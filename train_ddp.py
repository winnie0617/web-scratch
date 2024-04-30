import logging
import sys
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')
logger.info("Logger set up")

import dataloader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch

import multimodal

from transformers import AutoModelForCausalLM, AutoModel, AutoConfig
from transformers import Pix2StructForConditionalGeneration, Pix2StructVisionConfig

checkpoint = None

# DDP
dist.init_process_group("nccl")
rank = dist.get_rank()
print(f"Start running basic DDP example on rank {rank}.")
device_id = rank % torch.cuda.device_count()
device = torch.device(f'cuda:{device_id}')
torch.cuda.set_device(device=device)



# ### Config


from transformers import set_seed
set_seed(123)
num_examples = None
patch_height, patch_width = 32, 32
max_patches = 1000
image_encoder_path = "google/pix2struct-textcaps-base"
lm_path = "mistralai/Mistral-7B-v0.1"
torch.cuda.empty_cache()
    
# # Part 1: Preprocess Data
# Mark all pixels that belongs to the bounding boxes of positive candidates as targets

# ### Preprocess data

# In[3]:


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


# ### Generate prompt and label
# The full prompt is:
# 
# [patch embeddings] \n Based on the webpage screenshot, try to complete the following task:\n Task: [task] \n Previous actions:\n [actions] \n Which image patch contains the element to interact with next?"

# In[4]:


cols_to_remove = set(train_dataset.column_names)
cols_to_remove.remove("screenshot")
train_dataset = train_dataset.map(
    dataloader.get_prompt_target,
    batched=False,
    remove_columns=list(cols_to_remove)
)
train_dataset[2]


# In[5]:


# filter out those with bounding box out of range

# def box_in_range(example):
#     print(example)
#     l, b, _, _, = example["boxes"]
#     # width, height = example["screenshot"].size
#     width = height = 100
#     return l < width and b < height
    
# train_dataset = train_dataset.filter(lambda x: x, input_columns=['valid'])
# train_dataset = train_dataset.remove_columns(['valid'])
# train_dataset


# ### Tokenize Train Data

# In[6]:


from transformers import Pix2StructImageProcessor, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained(lm_path)
tokenizer.pad_token = tokenizer.eos_token # should be ok for casual LM
processor = Pix2StructImageProcessor.from_pretrained(image_encoder_path) # TODO: define this somewhere else
processor.max_patches = max_patches
processor.patch_size = {"height": patch_height, "width": patch_width}

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

# ### Prepare Model

# In[7]:


### Config for notebook
config = AutoConfig.from_pretrained("mistralai/Mistral-7B-v0.1")
config.return_dict = True
config.use_cache = False
config.low_cpu_mem_usage = True
config.rope_theta = 10000.0
config.attn_implementation = "flash_attention_2"
###
world_size = dist.get_world_size()

lm = AutoModelForCausalLM.from_pretrained(lm_path, config=config, torch_dtype=torch.bfloat16, device_map=device)

image_encoder_config = AutoConfig.from_pretrained(image_encoder_path)
image_encoder = Pix2StructForConditionalGeneration.from_pretrained(image_encoder_path, config=image_encoder_config).encoder
image_encoder.to(device)

model = multimodal.MultimodalAgent(config, image_encoder, lm, patch_width, patch_height)
model.to(device)
# print(f"Device: {device}:")
# print(torch.cuda.memory_allocated())

# print("Layers and their dimensions:")
# import torch.nn as nn
# for name, module in model.named_modules():
#     if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#         print(f"{name}: {module.weight.shape}")


# ### Set up LoRA

# In[8]:


from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training
lora_config = LoraConfig(
    # task_type=TaskType.CAUSAL_LM, # task type is not necessary, but this is needed to get the label
    inference_mode=False,
    r=16,
    lora_alpha=32, 
    lora_dropout=0.05,
    target_modules=['k_proj', 'wo', 'gate_proj', 'query', 'projector', 'q_proj', 'wi_1', 'down_proj', 'v_proj', 'wi_0', 'o_proj', 'key', 
'up_proj', 'output', 'value', 'patch_projection'], # exclude lm head
    modules_to_save=["projector"] # this layer is not pretrained
)

# model.lm.enable_input_require_grads()
model = get_peft_model(model, lora_config)

if checkpoint:
    # from peft import PeftModel
    # model = PeftModel.from_pretrained(model, checkpoint, config=lora_config)
    
    from peft import load_peft_weights, set_peft_model_state_dict
    lora_weights = load_peft_weights(checkpoint)
    set_peft_model_state_dict(model, lora_weights)

model.print_trainable_parameters()
# model = DDP(model, device_ids=[device_id])

# print("With checkpoint")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)

# print(model.peft_config["default"].target_modules)

# ### Set up Trainer

# In[9]:


from transformers import TrainingArguments

config = {
        'lora_config': lora_config,
        'learning_rate': 5e-4,
        'num_train_epochs': 30,
        'gradient_accumulation_steps': 128 / world_size,
        'per_device_train_batch_size': 1,
        'per_device_eval_batch_size': 1,
        'eval_accumulation_steps': 32,
        'gradient_checkpointing': True,
        'gradient_checkpointing_kwargs':{'use_reentrant':False}, # To work with DDP
        'ddp_find_unused_parameters': False
}


# ### Run Training

# In[10]:

eval_save_step = 60
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    optim="adamw_torch_fused",
    bf16=True,  # Use BF16 for flash attention
    # evlaution
    label_names=["labels"], # so that trainer will call compute_loss
    evaluation_strategy="steps",
    eval_steps=eval_save_step,
    include_inputs_for_metrics=True,
    log_level="info",
    # logging strategies
    logging_dir="output/logs",
    logging_strategy="steps",
    logging_first_step=True,
    logging_steps=20,
    # save model strategies
    save_strategy="steps",
    save_steps=eval_save_step,
    save_total_limit=1,
    remove_unused_columns=False,
    **{k:v for k,v in config.items() if k != 'lora_config'}
) # TODO: move train arguments to config
trainer = multimodal.MultimodalTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=multimodal.compute_metrics,
    tokenizer=tokenizer,
    data_collator=multimodal.custom_collate,
)
torch.cuda.empty_cache()
if not checkpoint:
    trainer.train()
else:
    trainer.train(resume_from_checkpoint=checkpoint)

dist.destroy_process_group()



