import hydra
import torch
import wandb
from dataloader import load
from transformers import (
    PreTrainedTokenizerFast,
    BartForConditionalGeneration,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # tokenizer
    # insert special token - special_tokens_map.json, tokenizer.json added_tokens, model-vocab
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.PATH.tokenizer)

    # model
    model = BartForConditionalGeneration.from_pretrained(cfg.MODEL.name)

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
