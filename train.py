import json
import logging
import pdb
import random
import tqdm

import hydra
import torch
from dataloader import get_data_split, convert_to_qa_format, preprocess_training_examples_with_tokenizer
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Use model {cfg.model.pretrained_model_name_or_path}")
    output_dir = HydraConfig.get().runtime.output_dir
    train_dataset = get_data_split(
        cfg.data.data_path, cfg.data.train_split_file, is_train=True
    )
    cols_to_remove = set(train_dataset.column_names)
    # keep clean_html
    cols_to_remove.remove("cleaned_html")
    train_dataset = train_dataset.filter(lambda x: len(x["cleaned_html"]) < 50000) # TODO: only for testing purpose
    train_dataset = train_dataset.map(
        convert_to_qa_format,
        batched=False,
        remove_columns=list(cols_to_remove)
    ).rename_column("cleaned_html", "context")
    
    # train_dataset = train_dataset.select(range(50))

    # for i in range(10):
    #     print(train_dataset[i]["answer"]["answer_start"])
    #     start, end = train_dataset[i]["answer"]["answer_start"][0], train_dataset[i]["answer"]["answer_end"][0]
    #     print("indexed:", train_dataset[i]["context"][start:end])
    #     print("actual:", train_dataset[i]["pos_candidates"][0])
    # print(train_dataset)
    # print(train_dataset[0]["answer"])
    
    # model = AutoModel.from_pretrained(cfg.model.pretrained_model_name_or_path, load_in_8bit=True, device_map="auto", use_cache=False) # TODO: hard coded
    print(cfg.model)
    print("============")
    model = AutoModelForCausalLM.from_pretrained(**cfg.model, torch_dtype=torch.bfloat16) # TODO: hard coded
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token # should be ok for casual LM

    # add new token
    # TODO: check again if this works properly after the code change
    tokenizer.add_tokens(["[ACT]"])
    model.resize_token_embeddings(len(tokenizer))
    params = model.state_dict()
    embeddings = params['model.embed_tokens.weight']
    pre_expansion_embeddings = embeddings[:-1,:]
    mu = torch.mean(pre_expansion_embeddings, dim=0)
    params['model.embed_tokens.weight'][-1,:] = mu
    model.load_state_dict(params)
    
    train_dataset = train_dataset.map(
    preprocess_training_examples_with_tokenizer(tokenizer, model.config.max_position_embeddings),
    # batched=True,
    # batch_size=256,
    batched=False,
    remove_columns=train_dataset.column_names,
    )
    
    # Calculate input length for each example
    input_lengths = []
    for example in train_dataset:
        input_lengths.append(len(example["input_ids"]))
    
    train_dataset.set_format("pt", columns=["input_ids", "attention_mask"], output_all_columns=True)
    logger.info(f"Use device {'gpu' if torch.cuda.is_available() else 'cpu'}")
    logger.info(f"Use batch size {cfg.train.batch_size}")
    logger.info(f"Training data size {len(train_dataset)}")
    
    # ==== start of new code
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, # task type is not necessary, but this is needed to get the label
        inference_mode=False,
        r=8,
        lora_alpha=16, 
        lora_dropout=0.05,
        # target_modules = ["q_proj", "v_proj"]
        target_modules = "all-linear"
    )

    # Dont int8 training
    # model = prepare_model_for_int8_training(model)
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.model.model.embed_tokens.weight.requires_grad = True
    model.print_trainable_parameters()
    
    # print all parameters that require grad
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    # Set up the trainer
    config = {
        'lora_config': lora_config,
        'learning_rate': cfg.train.learning_rate,
        'num_train_epochs': cfg.train.epoch,
        'gradient_accumulation_steps': cfg.train.gradient_accumulation_steps,
        'per_device_train_batch_size': cfg.train.batch_size,
        'gradient_checkpointing': True,
    }
    
    class CustomTrainer(Trainer):
        
        def compute_loss(self, model, inputs, return_outputs=False):

            # print(inputs["input_ids"][0][-1])
            # print(model.model.model.embed_tokens.weight[-1,:10])
            hidden_states = model(inputs["input_ids"], inputs["attention_mask"], output_hidden_states=True).hidden_states[0] # model_output.hidden_states is a tuple
            # print("Input length", inputs["input_ids"].shape)
            # print(torch.cuda.memory_summary())
            # act = hidden_states[:,-1,:]
            # pos = hidden_states[:,inputs["labels"][0]["pos_candidates"][0],:] # TODO: right now only using the first positive candidate and only works for batch size 1
            # compute cosine simularity between last token and every token before

            sim = torch.nn.functional.cosine_similarity(hidden_states[:,:-1,:], hidden_states[:,-1,:], dim=2)
            # TODO: add temperature?
            loss = torch.nn.functional.cross_entropy(sim, inputs["labels"])

            return loss
    
    def custom_collate(data):
        inputs = torch.stack([d['input_ids'] for d in data])
        attention_mask = torch.stack([d['attention_mask'] for d in data])
        labels = torch.tensor([d['label']["pos_candidates"][0] for d in data]) # todo: only uses first positive
        return { 
            'input_ids': inputs,
            'attention_mask': attention_mask,
            'labels': labels
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="no",
        optim="adamw_torch_fused",
        bf16=True,  # Use BF16 for flash attention
        # logging strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        **{k:v for k,v in config.items() if k != 'lora_config'}
    ) # TODO: move train arguments to config


    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=custom_collate,
    )

    trainer.train()

    # predictions, _, _ = trainer.predict(validation_dataset)
    # start_logits, end_logits = predictions
    # print(
    #     compute_metrics(
    #         start_logits, end_logits, validation_dataset, raw_datasets["validation"]
    #     )
    # )

if __name__ == "__main__":
    main()
