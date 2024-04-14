import logging
import sys
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')
logger.info("Logger set up")

import dataloader

from transformers import set_seed
set_seed(123)
num_examples = 3000
patch_height, patch_width = 16, 16
max_patches = 4000

from datasets import load_dataset

# logger.info(f"Use model {cfg.model.pretrained_model_name_or_path}")
# output_dir = HydraConfig.get().runtime.output_dir
if num_examples:
    train_dataset = load_dataset("osunlp/Multimodal-Mind2Web", split="train").select(range(num_examples))
else:
    train_dataset = load_dataset("osunlp/Multimodal-Mind2Web", split="train")
print(train_dataset)
train_dataset = train_dataset.remove_columns(["neg_candidates", "raw_html", "cleaned_html"])
train_dataset = dataloader.get_previous_actions(train_dataset)
# filter out those without pos_candidates
train_dataset = train_dataset.filter(lambda x: len(x)==1, input_columns=['pos_candidates'])
train_dataset = train_dataset.remove_columns(['action_reprs'])
print(train_dataset)

cols_to_remove = set(train_dataset.column_names)
cols_to_remove.remove("screenshot")
train_dataset = train_dataset.map(
    dataloader.get_prompt_target,
    batched=False,
    remove_columns=list(cols_to_remove)
)
train_dataset[2]

import multimodal

from transformers import AutoModelForCausalLM, AutoModel, AutoConfig
import torch
# from transformers import Pix2StructVisionModel, ViTImageProcessor, Pix2StructVisionConfig

### Config for notebook
config = AutoConfig.from_pretrained("mistralai/Mistral-7B-v0.1")
config.return_dict = True
config.use_cache = False
config.low_cpu_mem_usage = True
config.rope_theta = 10000.0
config.attn_implementation = "flash_attention_2"
###

# TODO: Move config to somewhere else

# image_encoder_config = Pix2StructVisionConfig.from_pretrained("google/pix2struct-base")
# TODO: try different hidden size?
# image_encoder_config.seq_len = 27145
# image_encoder_config.patch_size = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# image_encoder = Pix2StructVisionModel.from_pretrained("google/pix2struct-textcaps-base", config=image_encoder_config, torch_dtype=torch.bfloat16)
# image_encoder.to(device)

image_encoder_path = "google/vit-base-patch16-224"
image_encoder_config = AutoConfig.from_pretrained(image_encoder_path)
image_encoder = AutoModel.from_pretrained(image_encoder_path, config=image_encoder_config)
image_encoder.to(device)

lm_path = "mistralai/Mistral-7B-v0.1"
lm = AutoModelForCausalLM.from_pretrained(lm_path, config=config, torch_dtype=torch.bfloat16)
lm.to(device)

model = multimodal.MultimodalAgent(config, image_encoder, lm)
model.to(device)
print(torch.cuda.memory_allocated())

print("Layers and their dimensions:")
import torch.nn as nn
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        print(f"{name}: {module.weight.shape}")
        
from transformers import AutoImageProcessor, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(lm_path)
tokenizer.pad_token = tokenizer.eos_token # should be ok for casual LM
processor = AutoImageProcessor.from_pretrained(image_encoder_path)

cols = train_dataset.column_names
cols.remove("screenshot")
train_dataset = train_dataset.map(
    dataloader.get_tokenize_fn(tokenizer),
    remove_columns=cols,
    )
# train_dataset.set_format("pt", columns=["input_ids", "attention_mask", "label"], output_all_columns=True)
print(train_dataset[0])
train_dataset.set_transform(dataloader.get_preprocess_image_fn(processor, max_patches, patch_height, patch_width), output_all_columns=True) # process images on the fly
# split the train_dataset into train and validation
dataset = train_dataset.train_test_split(test_size=0.05) 
train_dataset, eval_dataset = dataset["train"], dataset["test"]
print(train_dataset[0])
logger.info(f"Use device {'gpu' if torch.cuda.is_available() else 'cpu'}")
# logger.info(f"Use batch size {cfg.train.batch_size}")
logger.info(f"Training data size {len(train_dataset)}")
logger.info(f"Eval data size {len(eval_dataset)}")

from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training
lora_config = LoraConfig(
    # task_type=TaskType.CAUSAL_LM, # task type is not necessary, but this is needed to get the label
    inference_mode=False,
    r=16,
    lora_alpha=32, 
    lora_dropout=0.05,
    target_modules="all-linear",
    modules_to_save=["projector"] # this layer is not pretrained
)

# model.lm.enable_input_require_grads()
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
        
from transformers import TrainingArguments, Trainer

config = {
        'lora_config': lora_config,
        'learning_rate': 1e-4,
        'num_train_epochs': 1,
        'gradient_accumulation_steps': 16,
        'per_device_train_batch_size': 1,
        'per_device_eval_batch_size': 1,
        'eval_accumulation_steps': 32,
        'gradient_checkpointing': True,
}

import math
import image_utils
    
class MultimodalTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        
        # hidden_states = model(flattened_patches=inputs["flattened_patches"], attention_mask_image=inputs["attention_mask_image"], input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        hidden_states = model(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"])
        # compute cosine simularity between last token and every token before
        temperature = 0.1 # TODO: hard coded
        # sim = torch.nn.functional.cosine_similarity(hidden_states[:,:-3,:], hidden_states[:,-1:,:], dim=2) # Last 3 tokens are "[", "ACT", "]"
        num_cols = inputs["pixel_values"].shape[-1] // patch_width
        num_rows = inputs["pixel_values"].shape[-2] // patch_width
        num_patches = num_cols * num_rows
        sim = torch.nn.functional.cosine_similarity(hidden_states[:,1:num_patches+1,:], hidden_states[:,-1:,:], dim=2) # Last 3 tokens are "[", "ACT", "]"
        pos_idxs = set()
        
        for box in inputs["labels"][0]: # TODO: only for batch size 
            pos_idxs.update(image_utils.boxes_to_patch_idx(box, num_cols, patch_width, patch_height))
        # +1 because first idx is CLS
        # target_idx = torch.tensor([idx + 1 for idx in pos_idxs]).to(device)

        target_idx = torch.tensor(list(pos_idxs)).to(device)

        if target_idx >= num_patches: # TODO: seems like some samples have bounding box that is out of range
            print("Bounding box out of range")
            target_idx = torch.tensor([0]).to(device)
            
        # if return_outputs and target_idx.item() in list(range(0, max_patches, 8)):
        if torch.argmax(sim).item() == target_idx.item():
            print(tokenizer.decode(inputs["input_ids"][0]))
            print("prediction", torch.argmax(sim).item(), "actual", target_idx.item())
            image_utils.plot_image(inputs["pixel_values"], patch_width, patch_height, torch.argmax(sim).item(), target_idx.item())
        
        # print("box", inputs["labels"][0])
        # print("click coordinate", patch_idx_to_click(target_idx, num_cols))
        # print("click box", patch_idx_to_patch_box(target_idx, num_cols))

        loss = torch.nn.functional.cross_entropy(sim / temperature, target_idx) # TODO: use BCE for multitarget?
        # print(loss)
        
        # print(torch.max(sim), sim[0,target_idx])
        if return_outputs:
            # instead of returning all hidden_states which would be too much memory,
            # return the similarity scores as "logits"
            # but different than sim because sin only calculates for 
            # scores = torch.nn.functional.cosine_similarity(hidden_states[:,:-1,:], hidden_states[:,-1:,:], dim=2)
            return loss, {"sim":sim, "target_idx":target_idx}
        return loss
    
import numpy as np
def custom_collate(data):
    # flattened_patches = torch.stack([d['screenshot'] for d in data])
    pixel_values = torch.stack([d['screenshot'] for d in data])
    # input_ids = torch.stack([d['input_ids'] for d in data])
    input_ids = torch.tensor([d['input_ids'] for d in data]) # set_transform resets set_format :(
    attention_mask = torch.tensor([d['attention_mask'] for d in data])
    # attention_mask_image = torch.stack([d['attention_mask_image'] for d in data])
    labels = torch.tensor([d['labels'] for d in data]) # todo: only uses first positive
    return { 
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        # 'attention_mask_image': attention_mask_image,
        'labels': labels,
    }

def compute_metrics(pred):
    sims, target_idxs = pred.predictions[0], pred.predictions[1]
    accuracy = []
    # Need to use a for loop because sequence length is different for each input
    preds = sims.argmax(axis=1)
    # print(np.max(sims, axis=1), [sims[r][i] for r, i in zip(range(len(target_idxs)), target_idxs)])
    accuracy = preds == target_idxs # TODO: use information from bounding box to get more metrics
    # bounding box stored in pred.label_ids
    return {
        'accuracy': np.array(accuracy).mean(),
    }
    
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    optim="adamw_torch_fused",
    bf16=True,  # Use BF16 for flash attention
    # evlaution
    label_names=["labels"], # so that trainer will call compute_loss
    evaluation_strategy="steps",
    eval_steps=20,
    include_inputs_for_metrics=True,
    log_level="info",
    # logging strategies
    logging_dir=f"output/logs",
    logging_strategy="steps",
    logging_steps=20,
    save_strategy="no",
    remove_unused_columns=False,
    **{k:v for k,v in config.items() if k != 'lora_config'}
) # TODO: move train arguments to config
trainer = MultimodalTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=custom_collate,
)
trainer.train()