import json
import pathlib
import pdb
import random
import re
import sys
from multiprocessing import Pool

from datasets import load_dataset
from transformers import AutoTokenizer

import lxml
from lxml import etree
from sentence_transformers import InputExample
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(pathlib.Path(__file__).parent.parent.absolute().as_posix())

from data_utils.dom_utils import get_tree_repr, prune_tree

def format_candidate(dom_tree, candidate, keep_html_brackets=False):
    node_tree = prune_tree(dom_tree, [candidate["backend_node_id"]])
    c_node = node_tree.xpath("//*[@backend_node_id]")[0]
    if c_node.getparent() is not None:
        c_node.getparent().remove(c_node)
        ancestor_repr, _ = get_tree_repr(
            node_tree, id_mapping={}, keep_html_brackets=keep_html_brackets
        )
    else:
        ancestor_repr = ""
    subtree_repr, _ = get_tree_repr(
        c_node, id_mapping={}, keep_html_brackets=keep_html_brackets
    )
    if subtree_repr.strip():
        subtree_repr = " ".join(subtree_repr.split()[:100])
    else:
        subtree_repr = ""
    if ancestor_repr.strip():
        ancestor_repr = re.sub(r"\s*\(\s*", "/", ancestor_repr)
        ancestor_repr = re.sub(r"\s*\)\s*", "", ancestor_repr)
        ancestor_repr = " ".join(ancestor_repr.split()[-50:])
    else:
        ancestor_repr = ""
    return f"ancestors: {ancestor_repr}\n" + f"target: {subtree_repr}"
    
    
def get_data_split(data_dir, split_file, is_train=False):
    def flatten_actions(samples):
        """ Creates one sample per action """
        outputs = {
            "website": [],
            "confirmed_task": [],
            "annotation_id": [],
            "previous_actions": [],
            "action_uid": [],
            "operation": [],
            "pos_candidates": [],
            # "neg_candidates": [], # Don't need neg_candidates in this case
            "cleaned_html": [],
        }
        num_actions = [len(actions) for actions in samples["actions"]]
        # Create number of sample per task = number of actions in the task
        for key in ["website", "confirmed_task", "annotation_id"]:
            for idx, value in enumerate(samples[key]):
                outputs[key] += [value] * num_actions[idx]
        for actions, action_reprs in zip(samples["actions"], samples["action_reprs"]):
            for a_idx, action in enumerate(actions):
                outputs["previous_actions"].append(action_reprs[:a_idx])
                for key in [
                    "action_uid",
                    "operation",
                    "pos_candidates",
                    # "neg_candidates", # Don't need neg_candidates in this case
                    "cleaned_html",
                ]:
                    outputs[key].append(action[key])
        return outputs

    dataset = load_dataset(data_dir, data_files=split_file, split="all")
    flatten_dataset = dataset.map(
        flatten_actions,
        batched=True,
        remove_columns=dataset.column_names, # remove all original columns?
        batch_size=10,
        num_proc=4,
    )

    def format_candidates(sample):
        dom_tree = lxml.etree.fromstring(sample["cleaned_html"])
        positive = []
        for candidate in sample["pos_candidates"]:
            positive.append(
                (
                    candidate["backend_node_id"],
                    format_candidate(dom_tree, candidate, keep_html_brackets=False),
                )
            )
        sample["pos_candidates"] = positive
        # negative = []
        # for candidate in sample["neg_candidates"]:
        #     negative.append(
        #         (
        #             candidate["backend_node_id"],
        #             format_candidate(dom_tree, candidate, keep_html_brackets=False),
        #         )
        #     )
        # sample["neg_candidates"] = negative
        return sample

    # flatten_dataset = flatten_dataset.map(
    #     format_candidates,
    #     num_proc=8,
    # )

    if is_train:
        flatten_dataset = flatten_dataset.filter(lambda x: len(x["pos_candidates"]) > 0)
    return flatten_dataset


# def format_input_generation(
#     sample, gt=-1, previous_k=5, keep_html_brackets=False):
#     """ Adapted from format_input_generation """
#     dom_tree = lxml.etree.fromstring(sample["cleaned_html"])
#     # TODO: Don't prune, just include the whole webpage
#     # dom_tree = prune_tree(dom_tree, candidate_ids)
#     tree_repr, id_mapping = get_tree_repr(
#         dom_tree, id_mapping={}, keep_html_brackets=keep_html_brackets
#     )
#     candidate_nodes = dom_tree.xpath("//*[@backend_node_id]")
#     # choices = []
#     # for idx, node in enumerate(candidate_nodes):
#     #     choices.append(
#     #         [
#     #             node.attrib["backend_node_id"],
#     #             " ".join(
#     #                 get_tree_repr(
#     #                     node,
#     #                     id_mapping=id_mapping,
#     #                     keep_html_brackets=keep_html_brackets,
#     #                 )[0].split()[:10]
#     #             ),
#     #         ]
#     #     )
#     gt = id_mapping.get(gt, -1)
#     seq_input = (
#         "Based on the HTML webpage above, try to complete the following task:\n"
#         f"Task: {sample['confirmed_task']}\n"
#         f"Previous actions:\n"
#     )
#     if len(sample["previous_actions"]) > 0:
#         for action in sample["previous_actions"][-previous_k:]:
#             seq_input += f"{action}\n"
#     else:
#         seq_input += "None\n"
#     seq_input += (
#         # "What should be the next action?"
#         # "Please select the element to interact with, and the action to perform along with the value to type in or select. "
#         # "If the task cannot be completed, output None."
#         "What should be the element to interact with next?"
#     )

#     if gt == -1:
#         seq_target = "None"
#     else:
#         current_action_op = sample["operation"]["op"]
#         current_action_value = sample["operation"]["value"]
#         seq_target = f"Element: {choices[gt][1]}\n"
#         seq_target += f"Action: {current_action_op}\n"
#         if current_action_op != "CLICK":
#             seq_target += f"Value: {current_action_value}"
#     return tree_repr, seq_input, seq_target, choices

def convert_to_qa_format(example):
    """ 
    Obtain the start and end char of the answer 
    Add columns "question", "context", "answer",
    Where answer is {"answer_start": idx, "answer_end": idx}
    """
    # dom_tree = lxml.etree.fromstring(example["cleaned_html"])
    #TODO: might be possible to have more than one pos candidate
    answer_start_idxs = []
    answer_end_idxs = []
    for pos_candidate in example["pos_candidates"]:
        pos_candidate_id = pos_candidate["backend_node_id"]
        id_attr = f'backend_node_id="{pos_candidate_id}"'
        idx = example["cleaned_html"].find(id_attr) # position of the 'b'
        element_end = idx - 2
        # open_bracket = element_end
        # while open_bracket >= 0 and example["cleaned_html"][open_bracket] != "<":
        #     open_bracket -= 1
        # element = example["cleaned_html"][open_bracket+1 : element_end+1] # eg li, a
        element = pos_candidate["tag"]
        
        # answer_start_idxs.append(open_bracket)
        # search for end of the tag
        close_bracket = element_end
        while close_bracket < len(example["cleaned_html"]) and example["cleaned_html"][close_bracket] != ">":
            close_bracket += 1
            
        # if "/" appears before ">", then no closing tag
        if example["cleaned_html"][close_bracket-1] == "/":
            answer_end_idxs.append(close_bracket)
        else:
            # scan until matching closing tag is found
            i = close_bracket
            counts = 1
            # TODO: handle out of index
            while i < len(example["cleaned_html"]) and counts > 0:
                if example["cleaned_html"][i] == "<":
                    if example["cleaned_html"][i+1] == "/" and example["cleaned_html"][i+2: i+2+len(element)] == element:
                        counts -= 1
                        i += len(element)
                        
                    elif example["cleaned_html"][i+1: i+1+len(element)] == element:
                        counts += 1
                        i += len(element)
                i += 1
            # print(example["cleaned_html"][element_end-len(element): i+2])
            # print("========")
            # i+1 is the position of the closing bracket
            if i == len(example["cleaned_html"]):
                print(i, pos_candidate, close_bracket)
            answer_end_idxs.append(i+1)

    # # NOTE: Don't prune, just include the whole webpage
    seq_input = (
        "Based on the HTML webpage, try to complete the following task:\n"
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
        # "What should be the next action?"
        # "Please select the element to interact with, and the action to perform along with the value to type in or select. "
        # "If the task cannot be completed, output None."
        "What should be the element to interact with next?"
    )

    # if gt == -1:
    #     seq_target = "None"
    # else:
    #     current_action_op = sample["operation"]["op"]
    #     current_action_value = sample["operation"]["value"]
    #     seq_target = f"Element: {choices[gt][1]}\n"
    #     seq_target += f"Action: {current_action_op}\n"
    #     if current_action_op != "CLICK":
    #         seq_target += f"Value: {current_action_value}"
    example["question"] = seq_input
    example["answers"] = {"answer_start": answer_start_idxs, "answer_end": answer_end_idxs}
    # example["context"] = example["cleaned_html"]
    return example

def preprocess_training_examples(examples, tokenizer, max_context_len):
    """
    Tokenize and break down long context
    """
    questions = [q.strip() for q in examples["question"]]
        
    inputs = tokenizer(
        examples["context"] + "\n" + examples["question"] + " [ACT]",
        return_offsets_mapping=True,
        # return_tensors="pt",
    )
    # print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))

    # determine the start and end positions of the answer
    
    # offset_mapping[i]: a tuple indicating the token i’s start position and end position of the span of characters inside the original context
    end_positions = []
    offset_mapping = inputs.pop("offset_mapping")
    for char_idx in examples["answers"]["answer_end"]:
        idx = len(offset_mapping) - 1
        while idx >= 0 and offset_mapping[idx][1] >= char_idx:
            idx -= 1
        end_positions.append(idx + 2)
        # print(char_idx, offset_mapping[idx + 2])
        # print(examples["context"][char_idx], inputs["input_ids"][idx + 2])

    # # sample_map = inputs.pop("overflow_to_sample_mapping") #  Since one sample can give several features, it maps each feature to the example it originated from
    # answers = examples["answers"]
    # start_positions = []
    # end_positions = []

    # offset is tuples of two integers representing the span of characters inside the original context
    # offset[i] gives a tuple of indices corresponding to the i-th token
    
    # for i, offset in enumerate(offset_mapping):
    #     sample_idx = sample_map[i]
    #     answer = answers[sample_idx]
    #     start_char = answer["answer_start"][0] # index
    #     end_char = answer["answer_end"][0] # end char is exclusive
    #     # They represent the index of the input sequence associated to each token. The sequence id can be None if the token is not related to any input sequence, like for example with special tokens.
    #     sequence_ids = inputs.sequence_ids(i)

        # # find the indices that start and end the context in the input IDs
        # idx = 0
        # while sequence_ids[idx] != 1:
        #     idx += 1
        # context_start = idx # found the first 1 mask
        # while sequence_ids[idx] == 1:
        #     idx += 1
        # context_end = idx - 1 # found the last 1 mask

    #     # If the answer is not fully inside the context, label is (0, 0)
    #     if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
    #         start_positions.append(0)
    #         end_positions.append(0)
    #     else:
    #         # Otherwise it's the start and end token positions (in the input ids)
    #         idx = context_start
    #         while idx <= context_end and offset[idx][0] <= start_char:
    #             idx += 1
    #         start_positions.append(idx - 1)

    #         idx = context_end
    #         while idx >= context_start and offset[idx][1] >= end_char:
    #             idx -= 1
    #         end_positions.append(idx + 1)

    # inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    
    # Set end token to be the ">" of the starting tag
    return inputs

def preprocess_training_examples_with_tokenizer(tokenizer, max_context_len):
    return lambda examples: preprocess_training_examples(examples, tokenizer, max_context_len)

class MultiChoiceDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        neg_ratio=5,
        num_candidates=5,
        max_context_len=512,
        mode="multichoice",
        top_k=-1,
    ):
        self.data = data
        self.neg_ratio = neg_ratio
        self.tokenizer = tokenizer
        self.num_candidates = num_candidates
        self.max_context_len = max_context_len
        self.mode = mode
        self.top_k = top_k

    def __len__(self):
        return len(self.data) * 10

    def __getitem__(self, idx):
        sample = self.data[idx // 10]
        if self.top_k > 0:
            top_negatives = [
                c for c in sample["neg_candidates"] if c["rank"] < self.top_k
            ]
            other_negatives = [
                c for c in sample["neg_candidates"] if c["rank"] >= self.top_k
            ]
        else:
            top_negatives = []
            other_negatives = sample["neg_candidates"]
        if random.random() < 0.8 and len(top_negatives) > 0:
            neg_candidates = top_negatives
        else:
            neg_candidates = other_negatives

        if len(sample["pos_candidates"]) != 0:
            # TODO: why need to randomly select candidate?
            pos_candidate = random.choice(sample["pos_candidates"])
            gt = pos_candidate["backend_node_id"] # id of the selected pos candidate
        else:
            gt = -1

        seq_context, seq_in, seq_out, _ = format_input_generation(
            sample, gt
        )

        seq_context = self.tokenizer(
            seq_context,
            truncation=True,
            max_length=self.max_context_len,
            add_special_tokens=False,
        )
        seq_in = self.tokenizer(
            seq_in,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_context_len,
        )
        model_input = {
            "input_ids": seq_context["input_ids"] + seq_in["input_ids"],
            "attention_mask": seq_context["attention_mask"] + seq_in["attention_mask"],
        }
        seq_out = self.tokenizer(seq_out)
        model_input["labels"] = seq_out["input_ids"]
        return model_input

