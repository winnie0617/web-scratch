from datasets import load_dataset
from transformers import AutoTokenizer
# from dataloader import CandidateRankDataset, get_data_split
import torch
from transformers import AutoModelForQuestionAnswering
import collections
import numpy as np
from tqdm.auto import tqdm
import evaluate


def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second", # truncate the context (which is in the second position) when the question with its context is too long
        stride=stride, # number of overlapping tokens between two successive chunks
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # determine the start and end positions of the answer
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


def compute_metrics(start_logits, end_logits, features, examples):
    # map each example in small_eval_set to the corresponding features in eval_set
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[
                            offsets[start_index][0] : offsets[end_index][1]
                        ],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in examples
    ]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


raw_datasets = load_dataset("squad")
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
max_length = 384
stride = 128

train_dataset = raw_datasets["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

validation_dataset = raw_datasets["validation"].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=raw_datasets["validation"].column_names,
)

small_eval_set = raw_datasets["validation"].select(range(100))
trained_checkpoint = "distilbert-base-cased-distilled-squad"

# Use a different tokenizer for preprocess_validation_examples
tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)
eval_set = small_eval_set.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=raw_datasets["validation"].column_names,
)

# Done with validation, revert back to original tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
eval_set_for_model.set_format("torch")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}
trained_model = AutoModelForQuestionAnswering.from_pretrained(trained_checkpoint).to(
    device
)

with torch.no_grad():
    outputs = trained_model(**batch)

start_logits = outputs.start_logits.cpu().numpy()
end_logits = outputs.end_logits.cpu().numpy()

n_best = 20
max_answer_length = 30
predicted_answers = []

metric = evaluate.load("squad")

print(compute_metrics(start_logits, end_logits, eval_set, small_eval_set))


# MODEL TRAINING
# from torch.utils.data import DataLoader
# from transformers import default_data_collator

# train_dataset.set_format("torch")
# validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
# validation_set.set_format("torch")

# train_dataloader = DataLoader(
#     train_dataset,
#     shuffle=True,
#     collate_fn=default_data_collator,
#     batch_size=8,
# )
# eval_dataloader = DataLoader(
#     validation_set, collate_fn=default_data_collator, batch_size=8
# )

# model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

# from torch.optim import AdamW

# optimizer = AdamW(model.parameters(), lr=2e-5)

# from accelerate import Accelerator

# accelerator = Accelerator(fp16=True)
# model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
#     model, optimizer, train_dataloader, eval_dataloader
# )

# from transformers import get_scheduler

# num_train_epochs = 3
# num_update_steps_per_epoch = len(train_dataloader)
# num_training_steps = num_train_epochs * num_update_steps_per_epoch

# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=0,
#     num_training_steps=num_training_steps,
# )

# from tqdm.auto import tqdm
# import torch

# progress_bar = tqdm(range(num_training_steps))

# for epoch in range(num_train_epochs):
#     # Training
#     model.train()
#     for step, batch in enumerate(train_dataloader):
#         outputs = model(**batch)
#         loss = outputs.loss
#         accelerator.backward(loss)

#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.update(1)

#     # Evaluation
#     model.eval()
#     start_logits = []
#     end_logits = []
#     accelerator.print("Evaluation!")
#     for batch in tqdm(eval_dataloader):
#         with torch.no_grad():
#             outputs = model(**batch)

#         start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
#         end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

#     start_logits = np.concatenate(start_logits)
#     end_logits = np.concatenate(end_logits)
#     start_logits = start_logits[: len(validation_dataset)]
#     end_logits = end_logits[: len(validation_dataset)]

#     metrics = compute_metrics(
#         start_logits, end_logits, validation_dataset, raw_datasets["validation"]
#     )
#     print(f"epoch {epoch}:", metrics)

#     # Save and upload
#     accelerator.wait_for_everyone() # make sure we have the same model in every process
#     unwrapped_model = accelerator.unwrap_model(model) # undo accerlate.prepare
#     unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
#     # if accelerator.is_main_process:
#     #     tokenizer.save_pretrained(output_dir)
#         # repo.push_to_hub(
#         #     commit_message=f"Training in progress epoch {epoch}", blocking=False
#         # )
        

        
# OLD TRAINING
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

from transformers import TrainingArguments

args = TrainingArguments(
    "bert-finetuned-squad",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=True,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
)
trainer.train()

predictions, _, _ = trainer.predict(validation_dataset)
start_logits, end_logits = predictions
print(
    compute_metrics(
        start_logits, end_logits, validation_dataset, raw_datasets["validation"]
    )
)
