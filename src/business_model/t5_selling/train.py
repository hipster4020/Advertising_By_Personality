import hydra
import torch
import wandb
from dataloader import load
from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(cfg.MODEL.name)

    # 스페셜 토큰 추가하기
    special_tokens = {"additional_special_tokens": ["<sep>", "<input>"]}
    tokenizer.add_special_tokens(special_tokens)

    # model
    model = T5ForConditionalGeneration.from_pretrained(cfg.MODEL.name)
    model.resize_token_embeddings(len(tokenizer))

    # data loder
    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)

    # wandb
    wandb.init(
        project=cfg.ETC.project,
        entity=cfg.ETC.entity,
        name=cfg.ETC.name,
    )

    # trainer
    args = TrainingArguments(
        **cfg.TRAININGARGS,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    trainer.train()

    trainer.save_model(cfg.PATH.save_dir)


if __name__ == "__main__":
    main()
