from datasets import load_dataset
from transformers import AutoTokenizer
# from dataloader import CandidateRankDataset, get_data_split
import torch
from transformers import AutoModelForQuestionAnswering
import collections
import numpy as np
from tqdm.auto import tqdm
import evaluate


class QAModel(nn.Module):
    
    def __init__(self, model_name_or_path):
        super().__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
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
