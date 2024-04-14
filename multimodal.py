from transformers import PreTrainedModel, Trainer
import torch
import torch.nn as nn
import image_utils
import numpy as np

class MultimodalAgent(PreTrainedModel):
    def __init__(self, config, image_encoder, lm, patch_width, patch_height):
        super().__init__(config)
        self.config = config
        self.supports_gradient_checkpointing = True
        self.image_encoder = image_encoder
        self.projector = nn.Linear(image_encoder.config.hidden_size, lm.config.hidden_size) 
        self.lm = lm
        self.patch_width = patch_width
        self.patch_height = patch_height

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        # embed pixel_values with image_encoder
        # h_image = self.image_encoder(flattened_patches, attention_mask_image).last_hidden_state
        h_image = self.image_encoder(pixel_values, interpolate_pos_encoding=True).last_hidden_state
        # linear layer to project hidden states to lm's input dimension
        h_image = self.projector(h_image)
        # look up token embedding for text
        h_text = self.lm.model.embed_tokens(input_ids)
        # concatenate image represenation with question
        inputs_embeds = torch.cat([h_image, h_text], dim=1)
        # also concat attention mask
        # attention_mask = torch.cat([torch.ones(h_image.shape), attention_mask], dim=-1)
        # TODO: need to add some sort of separator, like \n?
        return self.lm(inputs_embeds=inputs_embeds, output_hidden_states=True).hidden_states[-1] # Not passing attention mask, no need for now since batch size is 1
        

class MultimodalTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        
        # hidden_states = model(flattened_patches=inputs["flattened_patches"], attention_mask_image=inputs["attention_mask_image"], input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        hidden_states = model(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"])
        # compute cosine simularity between last token and every token before
        temperature = 0.1 # TODO: hard coded
        # sim = torch.nn.functional.cosine_similarity(hidden_states[:,:-3,:], hidden_states[:,-1:,:], dim=2) # Last 3 tokens are "[", "ACT", "]"
        num_cols = inputs["pixel_values"].shape[-1] // model.patch_width
        num_rows = inputs["pixel_values"].shape[-2] // model.patch_height
        num_patches = num_cols * num_rows
        sim = torch.nn.functional.cosine_similarity(hidden_states[:,1:num_patches+1,:], hidden_states[:,-1:,:], dim=2) # Last 3 tokens are "[", "ACT", "]"
        pos_idxs = set()
        
        for box in inputs["labels"][0]: # TODO: only for batch size 
            pos_idxs.update(image_utils.boxes_to_patch_idx(box, num_cols, model.patch_width, model.patch_height))
        # +1 because first idx is CLS
        # target_idx = torch.tensor([idx + 1 for idx in pos_idxs]).to(device)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # TODO: Move

        target_idx = torch.tensor(list(pos_idxs)).to(device)

        if target_idx >= num_patches: # TODO: seems like some samples have bounding box that is out of range
            print("Bounding box out of range")
            target_idx = torch.tensor([0]).to(device)
            
        # if return_outputs and target_idx.item() in list(range(0, max_patches, 8)):
        #     print(tokenizer.decode(inputs["input_ids"][0]))
        #     print("prediction", torch.argmax(sim).item(), "actual", target_idx.item())
        #     image_utils.plot_image(inputs["pixel_values"], patch_width, patch_height, torch.argmax(sim).item(), target_idx.item())
        
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