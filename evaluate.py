import json
import logging
import pdb
import random
import tqdm

import hydra
import torch
import numpy as np
from dataloader import convert_to_qa_format, preprocess_training_examples_with_tokenizer
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training
from transformers import set_seed

from datasets import load_dataset
from dataloader import prune_html, get_previous_actions

logger = logging.getLogger(__name__)

checkpoint = "output/checkpoint-150"


    
def custom_collate(data):
    inputs = torch.stack([d['input_ids'] for d in data])
    attention_mask = torch.stack([d['attention_mask'] for d in data])
    labels = torch.tensor([d['label'][0] for d in data]) # todo: only uses first positive
    return { 
        'input_ids': inputs,
        'attention_mask': attention_mask,
        'labels': labels
    }

def compute_metrics(pred):
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

        
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    logger.info(f"Use model {cfg.model.pretrained_model_name_or_path}")
    output_dir = HydraConfig.get().runtime.output_dir
    # dataset = get_data_split(
    #     cfg.data.data_path, cfg.data.train_split_file, is_train=True
    # )
    
    print(cfg.model)
    print("============")
    model = AutoModelForCausalLM.from_pretrained(**cfg.model, torch_dtype=torch.bfloat16) # TODO: hard coded
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token # should be ok for casual LM
    
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
            
            # if wrong prediction, write to file
            if target_idx != sim.argmax():
                pred = sim.argmax()
                with open("wrong_predictions.txt", "a") as f:
                    f.write(f"{tokenizer.decode(inputs['input_ids'][0])}\n")
                    f.write(f"Predicted: {sim.argmax()}, {tokenizer.decode(inputs['input_ids'][0][pred-20:pred+1])}\n")
                    f.write(f"Actual: {target_idx.item()}, {tokenizer.decode(inputs['input_ids'][0][target_idx.item()-20:target_idx.item()+1])}\n")
                    f.write("====================================\n")
            
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
    if checkpoint:
        # from peft import PeftModel
        # model = PeftModel.from_pretrained(model, checkpoint, config=lora_config)
        
        from peft import load_peft_weights, set_peft_model_state_dict
        lora_weights = load_peft_weights(checkpoint)
        set_peft_model_state_dict(model, lora_weights)
    
    model.print_trainable_parameters()
    
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
        logging_steps=5,
        # save model strategies
        save_strategy="steps",
        save_steps=10,
        save_total_limit=1,
        remove_unused_columns=False,
        **{k:v for k,v in config.items() if k != 'lora_config'}
    ) # TODO: move train arguments to config


    trainer = CustomTrainer(
        model=model,
        args=training_args,
        # dataset=dataset,
        # eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=custom_collate,
    )


    for split in ["test_task", "test_website", "test_domain"]:
        dataset = load_dataset(cfg.data.data_path, split=split) # TODO: only use 1000 for now
        cols_to_remove = set(dataset.column_names)
        
        print("Before any filtering")
        print(dataset)

        
        dataset = dataset.map(prune_html, batched=False) #, load_from_cache_file=False

        dataset = get_previous_actions(dataset)
        
        dataset = dataset.filter(lambda x: len(x)>=1, input_columns=['pos_candidates'])
        print("After filtering for empty pos_candidate")
        print(dataset)
        
        dataset = dataset.map(
            convert_to_qa_format,
            batched=False,
            remove_columns=list(cols_to_remove)
        )
        # filter data where answer is None
        
        dataset = dataset.filter(lambda x: x["answers"] != [])
        print("After filtering for elements that cannot be located")
        print(dataset)
        
        # dataset = dataset.filter(lambda x: len(x["context"]) < 60000)
        # print("More filtering for long context")
        # print(dataset)

        dataset = dataset.map(
            preprocess_training_examples_with_tokenizer(tokenizer, model.config.max_position_embeddings),
            # batched=True,
            # batch_size=256,
            batched=False,
            remove_columns=dataset.column_names,
            )
        
        # filter examples with too many tokens
        dataset = dataset.filter(lambda x: len(x["input_ids"]) < 30000)
        print("More filtering for long context")
        print(dataset)
        
        dataset.set_format("pt", columns=["input_ids", "attention_mask"], output_all_columns=True)
        # # split the dataset into train and validation
        
        logger.info(f"Testing on {split}")
        logger.info(f"Use device {'gpu' if torch.cuda.is_available() else 'cpu'}")
        logger.info(f"Test data size {len(dataset)}")
        
        eval_results = trainer.evaluate(dataset)
        print(eval_results)
        
if __name__ == "__main__":
    main()
