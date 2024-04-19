from transformers import PreTrainedModel, Trainer
import torch
import torch.nn as nn
import image_utils
import numpy as np
import time


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

    def forward(self, flattened_patches, input_ids, attention_mask, attention_mask_image, labels=None):
        # embed pixel_values with image_encoder
        h_image = self.image_encoder(flattened_patches, attention_mask_image).last_hidden_state
        # h_image = self.image_encoder(pixel_values, interpolate_pos_encoding=True).last_hidden_state
        # linear layer to project hidden states to lm's input dimension
        h_image = self.projector(h_image)
        
        # # use attention mask to keep only positions where attention_mask is 1
        # input_ids = input_ids[torch.where(attention_mask == 1)].unsqueeze(0)
        # print(input_ids)
        # look up token embedding for text
        input_ids_text = input_ids[torch.where(attention_mask == 1)].unsqueeze(0)
        input_ids_pads = input_ids[torch.where(attention_mask == 0)].unsqueeze(0)
        h_text = self.lm.model.embed_tokens(input_ids_text)
        h_pads = self.lm.model.embed_tokens(input_ids_pads)
        # concatenate image represenation with question
        inputs_embeds = torch.cat([h_pads, h_image, h_text], dim=1)
        
        # # also concat attention mask
        attention_mask_modified = torch.cat([attention_mask[:,:input_ids.shape[1]], attention_mask_image, attention_mask[:,input_ids.shape[1]:]], dim=-1)
        # TODO: need to add some sort of separator, like \n?
        return self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask_modified, output_hidden_states=True).hidden_states[-1] # Not passing attention mask, no need for now since batch size is 1        

class MultimodalTrainer(Trainer):
    
    
    def compute_loss(self, model, inputs, return_outputs=False):
        
        # with torch.autocast(device_type="cuda"):
        device = model.device
        hidden_states = model(flattened_patches=inputs["flattened_patches"], input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], attention_mask_image=inputs["attention_mask_image"])
        # compute cosine simularity between last token and every token before
        temperature = 0.1 # TODO: hard coded
        # sim = torch.nn.functional.cosine_similarity(hidden_states[:,:-3,:], hidden_states[:,-1:,:], dim=2) # Last 3 tokens are "[", "ACT", "]"
        
        # num_cols = inputs["pixel_values"].shape[-1] // model.patch_width
        num_rows, num_cols = inputs["row_col"][0]
        num_patches = num_cols * num_rows
        max_patches = inputs["attention_mask_image"].shape[-1]
        text_len = inputs["attention_mask"].sum()

        # max_patch = inputs["attention_mask_image"]
        
        # TODO: I believe no CLS token, but double check
        sim = torch.nn.functional.cosine_similarity(hidden_states[:,-(text_len+max_patches):-(text_len+max_patches-num_patches),:], hidden_states[:,-1:,:], dim=2) # Last 3 tokens are "[", "ACT", "]"
        # sim = torch.nn.functional.cosine_similarity(hidden_states[:,1:num_patches+1,:], hidden_states[:,-1:,:], dim=2) # Last 3 tokens are "[", "ACT", "]"
        
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # TODO: Move
        # get current cuda device
        
        model.gradient_as_bucket_view = True
        
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        
        target_idxs = []
        for boxes in inputs["labels"]:
            pos_idxs = set()
            for box in boxes: # TODO: only for batch size 
                pos_idxs.update(image_utils.boxes_to_patch_idx(box, num_cols, model.patch_width, model.patch_height))
            target_idxs.append(list(pos_idxs)[0])
        
        target_idxs = torch.tensor(target_idxs).to(device)
        
        if (target_idxs >= num_patches).sum() > 0: # TODO: seems like some samples have bounding box that is out of range
            print("Bounding box out of range")
            target_idxs[torch.where(target_idxs >= num_patches)] = 0
        
        target_idx = target_idxs.item()
        
        if return_outputs and target_idx in list(range(0, num_patches, 37)):
            # from transformers import AutoTokenizer
            # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
            # print(tokenizer.decode(inputs["input_ids"][0]))
            import matplotlib.pyplot as plt
            # print("prediction", torch.argmax(sim).item(), "actual", target_idx)
            plt.imshow(sim.cpu().detach().numpy().reshape(num_rows, num_cols))
            plt.colorbar()
            # configure colormap theme
            plt.set_cmap('gray')
            # draw a circle at target_idx
            plt.plot(target_idx % num_cols.item(), target_idx // num_cols.item(), 'go')
            plt.plot(torch.argmax(sim).item() % num_cols.item(), torch.argmax(sim).item() // num_cols.item(), 'ro')
            # plt.show()
            plt.savefig(f'visualization/similarity/{target_idx}_{time.time()}.png')
            plt.clf()
            pixel_values = image_utils.flattened_patches_to_pixel_values(inputs["flattened_patches"][0], num_rows, num_cols, model.patch_width, model.patch_height)
            image_utils.plot_image(pixel_values, model.patch_width, model.patch_height, torch.argmax(sim).item(), target_idx, save_name=target_idx)
            plt.clf()
        
        # print("box", inputs["labels"][0])
        # print("click coordinate", patch_idx_to_click(target_idx, num_cols))
        # print("click box", patch_idx_to_patch_box(target_idx, num_cols))

        loss = torch.nn.functional.cross_entropy(sim / temperature, target_idxs) # TODO: use BCE for multitarget?
        # print(loss)
        
        # plot heat map for sim

        # print(torch.max(sim), sim[0,target_idx])
        if return_outputs:
            # instead of returning all hidden_states which would be too much memory,
            # return the similarity scores as "logits"
            # but different than sim because sin only calculates for 
            # scores = torch.nn.functional.cosine_similarity(hidden_states[:,:-1,:], hidden_states[:,-1:,:], dim=2)
            return loss, {"sim":sim, "target_idx":target_idxs}
        return loss


def custom_collate(data):
    # flattened_patches = torch.stack([d['screenshot'] for d in data])
    flattened_patches = torch.stack([d['screenshot'] for d in data])
    # input_ids = torch.stack([d['input_ids'] for d in data])
    input_ids = torch.tensor([d['input_ids'] for d in data]) # set_transform resets set_format :(
    attention_mask = torch.tensor([d['attention_mask'] for d in data])
    attention_mask_image = torch.stack([d['attention_mask_image'] for d in data])
    labels = torch.tensor([d['labels'] for d in data]) # todo: only uses first positive
    row_col = torch.tensor([d['row_col'] for d in data])
    return { 
        'flattened_patches': flattened_patches,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'attention_mask_image': attention_mask_image,
        'labels': labels,
        'row_col': row_col
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