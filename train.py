import json
import logging
import pdb
import random
import tqdm

import hydra
import torch
import numpy as np
from dataloader import get_data_split, convert_to_qa_format, preprocess_training_examples_with_tokenizer
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training
from transformers import set_seed


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    logger.info(f"Use model {cfg.model.pretrained_model_name_or_path}")
    output_dir = HydraConfig.get().runtime.output_dir
    train_dataset = get_data_split(
        cfg.data.data_path, cfg.data.train_split_file, is_train=True
    )
    cols_to_remove = set(train_dataset.column_names)
    # keep clean_html
    cols_to_remove.remove("cleaned_html")
    train_dataset = train_dataset.filter(lambda x: len(x["cleaned_html"]) < 50000) # TODO: 70000
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
    
    print(cfg.model)
    print("============")
    model = AutoModelForCausalLM.from_pretrained(**cfg.model, torch_dtype=torch.bfloat16) # TODO: hard coded
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token # should be ok for casual LM

    # add new token
    # TODO: check again if this works properly after the code change
    # tokenizer.add_tokens(["[ACT]"])
    # model.resize_token_embeddings(len(tokenizer))
    # params = model.state_dict()
    # embeddings = params['model.embed_tokens.weight']
    # pre_expansion_embeddings = embeddings[:-1,:]
    # mu = torch.mean(pre_expansion_embeddings, dim=0)
    # params['model.embed_tokens.weight'][-1,:] = mu
    # model.load_state_dict(params)
    
    train_dataset = train_dataset.map(
    preprocess_training_examples_with_tokenizer(tokenizer, model.config.max_position_embeddings),
    # batched=True,
    # batch_size=256,
    batched=False,
    remove_columns=train_dataset.column_names,
    )
    
    train_dataset.set_format("pt", columns=["input_ids", "attention_mask"], output_all_columns=True)
    # split the train_dataset into train and validation
    dataset = train_dataset.train_test_split(test_size=0.05) 
    train_dataset, eval_dataset = dataset["train"], dataset["test"]
    
    logger.info(f"Use device {'gpu' if torch.cuda.is_available() else 'cpu'}")
    logger.info(f"Use batch size {cfg.train.batch_size}")
    logger.info(f"Training data size {len(train_dataset)}")
    logger.info(f"Eval data size {len(eval_dataset)}")
    
    # ==== start of new code
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, # task type is not necessary, but this is needed to get the label
        inference_mode=False,
        r=16,
        lora_alpha=32, 
        lora_dropout=0.05,
        # target_modules = ["q_proj", "v_proj"]
        target_modules = "all-linear"
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    # model.model.model.embed_tokens.weight.requires_grad = True
    model.print_trainable_parameters()

    # Set up the trainer
    config = {
        'lora_config': lora_config,
        'learning_rate': cfg.train.learning_rate,
        'num_train_epochs': cfg.train.epoch,
        'gradient_accumulation_steps': cfg.train.gradient_accumulation_steps,
        'per_device_train_batch_size': cfg.train.batch_size,
        'per_device_eval_batch_size': cfg.eval.eval_batch_size,
        'eval_accumulation_steps': cfg.eval.eval_accumulation_steps,
        'gradient_checkpointing': True,
    }
    
    class CustomTrainer(Trainer):
        
        def compute_loss(self, model, inputs, return_outputs=False):

            # print(model.model.model.embed_tokens.weight[-1,:10])
            hidden_states = model(inputs["input_ids"], inputs["attention_mask"], output_hidden_states=True).hidden_states[-1]
            # act = hidden_states[:,-1,:]
            # pos = hidden_states[:,inputs["labels"][0]["pos_candidates"][0],:] # TODO: right now only using the first positive candidate and only works for batch size 1
            # compute cosine simularity between last token and every token before
            temperature = 0.1 # TODO: hard coded
            sim = torch.nn.functional.cosine_similarity(hidden_states[:,:-3,:], hidden_states[:,-1:,:], dim=2) # Last 3 tokens are "[", "ACT", "]"
            # sim = torch.nn.functional.cosine_similarity(hidden_states[:,:-1,:], hidden_states[:,-1:,:], dim=2)
            target_idx = inputs["labels"]
            
            # # get indices where input id is 28767 (">") or 2720 ("/>")
            # is_close = (inputs["input_ids"] == 28767) + (inputs["input_ids"] == 2720)
            # idxs = is_close.nonzero() - 1 # to get index before
            # # print(hidden_states[:,idxs[:,1][:5],:10])
            # sim = torch.nn.functional.cosine_similarity(hidden_states[:,idxs[:,1],:], hidden_states[:,-1,:], dim=2)
            # # map label to corresponding index in the subset of hidden_states
            # target_idx = is_close[:,:inputs["labels"]].sum(dim=1)

            loss = torch.nn.functional.cross_entropy(sim / temperature, target_idx)

            if return_outputs:
                # instead of returning all hidden_states which would be too much memory,
                # return the similarity scores as "logits"
                # but different than sim because sin only calculates for 
                # scores = torch.nn.functional.cosine_similarity(hidden_states[:,:-1,:], hidden_states[:,-1:,:], dim=2)
                return loss, {"similarity": sim}
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

    def compute_metrics(pred):
        # print("pred", pred.predictions.shape, pred.predictions)
        # find the first index where inputs is -100
        idx = (pred.inputs != -100).sum(axis=1) # if N tokens, then first -100 is at N
        accuracy = []
        number_max = []
        # Need to use a for loop because sequence length is different for each input
        for i in range(len(pred.inputs)):
            sim = pred.predictions[i,:idx[i]]
            max_sim = sim.max()
            is_max = sim == max_sim

            accuracy.append(is_max[pred.label_ids[i]])
            number_max.append(is_max.sum())

    
        return {
            'accuracy': np.array(accuracy).mean(),
            'number_max': np.array(number_max).mean(),
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        optim="adamw_torch_fused",
        bf16=True,  # Use BF16 for flash attention
        # evlaution
        evaluation_strategy="steps",
        eval_steps=cfg.eval.eval_steps,
        include_inputs_for_metrics=True,
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
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=custom_collate,
    )

    trainer.train()

if __name__ == "__main__":
    main()
