import json
import logging
import pdb
import random
import tqdm

import hydra
import torch
from dataloader import get_data_split, convert_to_qa_format, preprocess_training_examples_with_tokenizer
from hydra.core.hydra_config import HydraConfig
# from model import CrossEncoder
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer, AutoModelForQuestionAnswering, AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Use model {cfg.model.model_name_or_path}")
    output_dir = HydraConfig.get().runtime.output_dir
    train_dataset = get_data_split(
        cfg.data.data_path, cfg.data.train_split_file, is_train=True
    )
    cols_to_remove = set(train_dataset.column_names)
    # keep clean_html
    cols_to_remove.remove("cleaned_html")
    # train_dataset = CandidateRankDataset(train_data)
    train_dataset = train_dataset.map(
        convert_to_qa_format,
        batched=False,
        remove_columns=list(cols_to_remove)
    ).rename_column("cleaned_html", "context").select(range(100))
    
    # for i in range(10):
    #     print(train_dataset[i]["answer"]["answer_start"])
    #     start, end = train_dataset[i]["answer"]["answer_start"][0], train_dataset[i]["answer"]["answer_end"][0]
    #     print("indexed:", train_dataset[i]["context"][start:end])
    #     print("actual:", train_dataset[i]["pos_candidates"][0])
    # print(train_dataset)
    # print(train_dataset[0]["answer"])
    
    # model = AutoModelForQuestionAnswering.from_pretrained(cfg.model.model_name_or_path)
    model = AutoModel.from_pretrained(cfg.model.model_name_or_path, torch_dtype=torch.bfloat16) # TODO: hard coded
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    
    # set pad token to eos token
    tokenizer.pad_token = tokenizer.eos_token # TODO: would this cause problems?
    
    # add new token
    tokenizer.add_tokens(["[ACT]"])
    params = model.state_dict()
    embeddings = params['embed_tokens.weight']
    pre_expansion_embeddings = embeddings[:-1,:]
    mu = torch.mean(pre_expansion_embeddings, dim=0)
    params['embed_tokens.weight'][-1,:] = mu
    model.load_state_dict(params)
    
    train_dataset = train_dataset.map(
    preprocess_training_examples_with_tokenizer(tokenizer, model.config.max_position_embeddings),
    # batched=True,
    # batch_size=256,
    batched=False, #TODO: just don't batch?
    remove_columns=train_dataset.column_names,
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=cfg.train.batch_size
    )

    logger.info(f"Use device {'gpu' if torch.cuda.is_available() else 'cpu'}")
    logger.info(f"Use batch size {cfg.train.batch_size}")
    logger.info(f"Training data size {len(train_dataset)}")
    
    # ==== start of new code
    
    def get_loss(model_output):
        """
        Compute triplet loss
        """
        # get model embeddings
        print(model_output)
        print(model_output["hidden_states"])
    
    def parameters_to_fine_tune(model, mode):
        # TODO: always finetune all parameters?
        return model.parameters()
        
    
    def finetune(model, mode, dataset, batch_size=8, grad_accum=8):
        # x, y = add_prefixes(x, y, dataset)
        
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(DEVICE)

        optimizer = torch.optim.Adam(parameters_to_fine_tune(model, mode), lr=2e-5)
        # all_both = tokenize_gpt2_batch(tok, x, y)
        max_n = len(dataset) # TODO: change this
        pbar = tqdm.tqdm(range(max_n))
        idxs = []
        for step in pbar:
            model.train()

            if len(idxs) < batch_size // grad_accum:
                idxs = list(range(len(x)))
                random.shuffle(idxs)
            batch_idxs = idxs[: batch_size // grad_accum]
            idxs = idxs[batch_size // grad_accum :]

            # 1. Sample a random minibatch of examples of size batch_size // grad_accum using the batch_idxs variable
            # 2. Tokenize the batch using the tokenize_gpt2_batch function you implemented
            # batch_x = [x[i] for i in batch_idxs]
            # batch_y = [y[i] for i in batch_idxs]
            # combined_sequences = tokenize_gpt2_batch(tok, batch_x, batch_y)
            # Note: the ** operator will unpack a dictionary into keyword arguments to a function (such as your model)
            model_output = model(**dataset[batch_idxs], use_cache=False)
            # 3. Run the model on the batch, get the logits, and compute the loss using the get_loss function you implemented
            loss = get_loss(model_output) / grad_accum
            print("Compare", model_output.loss / grad_accum, loss)

            loss.backward()
            
            if (step+1) % grad_accum == 0: # don't want to take a step at step 0
                optimizer.step()
                optimizer.zero_grad()
            
            # END YOUR CODE

            if step % (grad_accum * 5) == 0:
                with torch.inference_mode():
                    model.eval()
                    accs = []
                    for idx in range(len(list(all_both.values())[0])):
                        d = {k: v[idx : idx + 1] for k, v in all_both.items()}
                        acc = get_acc(model(**d).logits, d["labels"])
                        accs.append(acc)
                    total_acc = sum(accs) / len(accs)
                    pbar.set_description(f"Fine-tuning acc: {total_acc:.04f}")

                if total_acc >= utils.early_stop_thresold(dataset):
                    print("Early stopping!")
                    break
        return model
    
    finetune(model, "train", train_dataset)
    # ===== original training code

    args = TrainingArguments(
        cfg.model.model_name_or_path,
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        # eval_dataset=validation_dataset,
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

    
    
    # model = CrossEncoder(
    #     cfg.model.model_name_or_path,
    #     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    #     num_labels=1,
    #     max_length=cfg.model.max_seq_length,
    # )

    # warmup_steps = int(len(train_dataloader) * cfg.train.warmup_steps)
    # logger.info(f"Warmup steps {warmup_steps}")

    # model.fit(
    #     optimizer_params={"lr": cfg.train.learning_rate},
    #     train_dataloader=train_dataloader,
    #     epochs=cfg.train.epoch,
    #     use_amp=cfg.train.use_amp,
    #     warmup_steps=warmup_steps,
    #     output_path=output_dir,
    # )
    # model.save(output_dir)


if __name__ == "__main__":
    main()
