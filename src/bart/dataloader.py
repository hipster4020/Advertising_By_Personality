from os.path import abspath, splitext
from typing import Optional

from datasets import load_dataset, logging

logging.set_verbosity(logging.ERROR)


def load(
    tokenizer,
    seq_len,
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    train_test_split: Optional[float] = None,
    worker: int = 1,
    batch_size: int = 1000,
    shuffle_seed: Optional[int] = None,
):
    def _tokenize_function(e):
        result = dict()
        decoder_input = dict()
        label = dict()

        result = tokenizer(
            [
                "<season>"
                + t1
                + "<ctrl1>"
                + t2
                + "<ctrl2>"
                + t3
                + tokenizer.sep_token
                + t4
                for t1, t2, t3, t4 in zip(
                    e["season"], e["ctrl1"], e["ctrl2"], e["input"]
                )
            ],
            max_length=seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        decoder_input = tokenizer(
            [tokenizer.bos_token + t for t in e["label"]],
            max_length=seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        label = tokenizer(
            [t + tokenizer.eos_token for t in e["label"]],
            max_length=seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        result["decoder_input_ids"] = decoder_input["input_ids"]
        result["decoder_attention_mask"] = decoder_input["attention_mask"]
        result["labels"] = label["input_ids"]

        return result

    train_data_path = abspath(train_data_path)
    is_eval = False
    _, extention = splitext(train_data_path)

    datafiles = {"train": train_data_path}
    if eval_data_path is not None:
        assert (
            train_test_split is None
        ), "Only one of eval_data_path and train_test_split must be entered."
        datafiles["test"] = abspath(eval_data_path)
        is_eval = True

    if train_test_split is not None:
        assert (
            0.0 < train_test_split < 1.0
        ), "train_test_split must be a value between 0 and 1"
        train_test_split = int(train_test_split * 100)
        train_test_split = {
            "train": f"train[:{train_test_split}%]",
            "test": f"train[{train_test_split}%:]",
        }
        is_eval = True

    data = load_dataset(
        extention.replace(".", ""),
        data_files=datafiles,
        split=train_test_split,
    )

    if shuffle_seed is not None:
        data = data.shuffle(seed=shuffle_seed)

    data = data.map(
        _tokenize_function,
        batched=True,
        batch_size=batch_size,
        num_proc=worker,
        remove_columns=data["train"].column_names,
    )

    return data["train"], (data["test"] if is_eval else None)


# Write preprocessor code to run in batches.
def default_collator(data):
    return data
