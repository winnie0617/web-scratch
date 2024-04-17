import json
import math

def get_previous_actions_original(dataset):
    
    # Add column for previous_actions
    previous_actions = []
    curr_actions = None
    num_actions = 0
    step = 0
    for i in range(len(dataset)):    
        if step == num_actions:
            step = 0
            curr_actions = dataset[i]["action_reprs"]
            num_actions = len(curr_actions)
        previous_actions.append(curr_actions[:step]) 
        step += 1

    dataset = dataset.add_column("previous_actions", previous_actions)
    
    return dataset

def get_previous_actions(dataset):
    
    # Add column for previous_actions
    previous_actions = []
    curr_actions = None
    num_actions = 0
    step = 0
    for i in range(len(dataset)):    
        if step == num_actions:
            step = 0
            curr_actions = dataset[i]["action_reprs"]
            num_actions = len(curr_actions)
        previous_actions.append(curr_actions[step]) 
        step += 1

    dataset = dataset.add_column("previous_actions", previous_actions)
    
    return dataset

def get_prompt_target_previous_action(example):
    """
    Use the bounding boxes of pos_candidates (as list of lists, [left, bottom, width, height]
    """
    boxes = []
    for cand in example["pos_candidates"]:
        json_data = json.loads(cand)
        attributes = json.loads(json_data['attributes'])
        bounding_box_rect_str = attributes['bounding_box_rect']
        boxes.append(list(map(float, bounding_box_rect_str.split(','))))

    # NOTE: Don't prune, just include the whole webpage
    seq_input = (
        "Based on the webpage screenshot, try to complete the following task:\n"
        f"Task: {example['confirmed_task']}\n"
        f"Previous actions:\n"
    )
    # TODO: hard-coded
    previous_k = 5
    if len(example["previous_actions"]) > 0:
        for action in example["previous_actions"][-previous_k:]:
            seq_input += f"{action}\n"
    else:
        seq_input += "None\n"
        
    seq_input += (
        "What should be the element to interact with next?"
    )

    example["question"] = seq_input
    example["boxes"] = boxes
    
    # l, b, _, _, = example["boxes"][0] # TODO: only works for single target
    # width, height = example["screenshot"].size
    # example["valid"] = l < width and b < height

    return example

def get_prompt_target(example):
    """
    Use the bounding boxes of pos_candidates (as list of lists, [left, bottom, width, height]
    """
    boxes = []
    for cand in example["pos_candidates"]:
        json_data = json.loads(cand)
        attributes = json.loads(json_data['attributes'])
        bounding_box_rect_str = attributes['bounding_box_rect']
        boxes.append(list(map(float, bounding_box_rect_str.split(','))))

    # NOTE: Don't prune, just include the whole webpage
    seq_input = (
        "Based on the webpage screenshot, try to complete the following task:\n"
        f"Task: {example['confirmed_task']}\n"
        f"Required actions:{example['previous_actions']}\n"
    )
    seq_input += (
        "What should be the element to interact with given the required action?"
    )

    example["question"] = seq_input
    example["boxes"] = boxes
    
    # l, b, _, _, = example["boxes"][0] # TODO: only works for single target
    # width, height = example["screenshot"].size
    # example["valid"] = l < width and b < height

    return example

def _tokenize_training_examples(examples, tokenizer):
    """
    Tokenize and map char index of the target to token index
    """
    inputs = tokenizer(examples["question"] + " [ACT]", padding="max_length", truncation=True, max_length=256)
    inputs["labels"] = examples["boxes"]

    return inputs

def get_tokenize_fn(tokenizer):
    return lambda examples: _tokenize_training_examples(examples, tokenizer)

def _preprocess_image(example, processor, max_patches, patch_height, patch_width):
    """ 
    Aspect ratio preserving, fixed size patches 
    reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/pix2struct/image_processing_pix2struct.py
    """
    
    image_width, image_height = example["screenshot"][0].size
    # maximize scale s.t.
    scale = math.sqrt(max_patches * (patch_height / image_height) * (patch_width / image_width))
    num_feasible_rows = max(min(math.floor(scale * image_height / patch_height), max_patches), 1)
    num_feasible_cols = max(min(math.floor(scale * image_width / patch_width), max_patches), 1)
    resized_height = max(num_feasible_rows * patch_height, 1)
    resized_width = max(num_feasible_cols * patch_width, 1)
    
    # processor.size = {"height":resized_height, "width":resized_width}
    inputs = processor(images=example["screenshot"], return_tensors="pt")
    example["screenshot"] = inputs["flattened_patches"]
    # example["screenshot"] = inputs["pixel_values"]
    all_scaled_boxes = []
    x_scale = image_width / resized_width
    y_scale = image_height / resized_height
    for boxes in example["labels"]:
        scaled_boxes = []
        for box in boxes:
            scaled_boxes.append([box[0]/x_scale, box[1]/y_scale, box[2]/x_scale, box[3]/y_scale])
        all_scaled_boxes.append(scaled_boxes)
    example["labels"] = all_scaled_boxes
    example["attention_mask_image"] = inputs["attention_mask"]
    example["row_col"] = [[num_feasible_rows, num_feasible_cols]]
    return example
    # return {"pixel_values": processor(images=example["screenshot"], return_tensors="pt").pixel_values} #[1, 3, 224, 224]
    
    
def _preprocess_image_cropped(example, processor, max_patches, patch_height, patch_width):
    """ 
    Aspect ratio preserving, fixed size patches and fixed resolution, crop image instead
    reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/pix2struct/image_processing_pix2struct.py
    """
    image_width, image_height = example["screenshot"][0].size
    scale = 1
    # Divide the image horizontally, choose the slice that contains the element
    # for patch_width x patch_height pixel patches, want <=max_patches while keeping original width
    # first calculate number of columns, and find the max number of rows for each slice
    num_cols = max(min(math.floor(scale * image_width / patch_width), max_patches), 1)
    num_rows_per_slice = math.floor(max_patches / num_cols)
    num_pixels_per_row = num_rows_per_slice * patch_height

    resized_height = max(int(image_height * scale // patch_height) * patch_height, 1)
    resized_width = max(num_cols * patch_width, 1)
    
    # processor.size = {"height":resized_height, "width":resized_width}
    inputs = processor(images=example["screenshot"], return_tensors="pt")
    # example["screenshot"] = inputs["flattened_patches"]
    
    all_scaled_boxes = []
    x_scale = image_width / resized_width
    y_scale = image_height / resized_height
    
    for boxes in example["labels"]:
        scaled_boxes = []
        for box in boxes:
            # Choose the slice that contains the box
            row_idx = int(box[1]/y_scale // num_pixels_per_row)
            pixel_idx = row_idx * num_pixels_per_row
            if pixel_idx >= image_height: # TODO: dont handle it here
                pixel_idx = 0
            scaled_boxes.append([box[0]/x_scale, box[1]/y_scale - pixel_idx, box[2]/x_scale, box[3]/y_scale]) # shift index to start at slice
        all_scaled_boxes.append(scaled_boxes)
    
    example["screenshot"] = inputs["pixel_values"][:, :, pixel_idx:pixel_idx+num_pixels_per_row, :]
    example["labels"] = all_scaled_boxes
    # print("height", resized_height, "width", resized_width)
    # print("box", box[1]/y_scale)
    # print(example["screenshot"][0].shape)
    # print("pixel_idx", pixel_idx, "num_pixels_per_row", num_pixels_per_row)
    
    return example

def get_preprocess_image_fn(processor, max_patches, patch_height, patch_width):
    return lambda example: _preprocess_image(example, processor, max_patches, patch_height, patch_width)
