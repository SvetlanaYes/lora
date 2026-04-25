from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from .config import DataConfig


def prepare_splits(dataset, data_config: DataConfig):
    splits = set(dataset.keys())

    if {"train", "validation", "test"}.issubset(splits):
        return dataset

    if {"train", "validation"}.issubset(splits) and "test" not in splits:
        validation_split = dataset["validation"].train_test_split(
            test_size=0.5,
            seed=data_config.split_seed,
        )
        dataset["validation"] = validation_split["train"]
        dataset["test"] = validation_split["test"]
        return dataset

    if {"train", "test"}.issubset(splits) and "validation" not in splits:
        train_split = dataset["train"].train_test_split(
            test_size=data_config.validation_fraction,
            seed=data_config.split_seed,
        )
        dataset["train"] = train_split["train"]
        dataset["validation"] = train_split["test"]
        return dataset

    if "train" in splits and "validation" not in splits and "test" not in splits:
        first_split = dataset["train"].train_test_split(
            test_size=data_config.validation_fraction + data_config.test_fraction,
            seed=data_config.split_seed,
        )
        holdout = first_split["test"].train_test_split(
            test_size=data_config.test_fraction / (
                data_config.validation_fraction + data_config.test_fraction
            ),
            seed=data_config.split_seed,
        )
        dataset["train"] = first_split["train"]
        dataset["validation"] = holdout["train"]
        dataset["test"] = holdout["test"]
        return dataset

    raise ValueError(
        f"Unsupported dataset split configuration for {data_config.dataset_name}: {sorted(splits)}"
    )


def tokenize_dataset(tokenizer, data_config: DataConfig):
    load_kwargs = {"trust_remote_code": True}
    if data_config.dataset_config_name:
        dataset = load_dataset(
            data_config.dataset_name,
            data_config.dataset_config_name,
            **load_kwargs,
        )
    else:
        dataset = load_dataset(data_config.dataset_name, **load_kwargs)
    dataset = prepare_splits(dataset, data_config)

    def tokenize_function(examples):
        return tokenizer(
            examples[data_config.dataset_text_field],
            truncation=True,
            padding=False,
            max_length=data_config.max_length,
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[data_config.dataset_text_field],
    )
    tokenized = tokenized.rename_column(
        data_config.dataset_label_field,
        "labels",
    )
    tokenized.set_format("torch")
    return tokenized


def build_dataloaders(tokenized_dataset, tokenizer, data_config: DataConfig):
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return {
        "train": DataLoader(
            tokenized_dataset[data_config.train_split],
            shuffle=True,
            batch_size=data_config.batch_size,
            collate_fn=collator,
        ),
        "validation": DataLoader(
            tokenized_dataset[data_config.validation_split],
            batch_size=data_config.batch_size,
            collate_fn=collator,
        ),
        "test": DataLoader(
            tokenized_dataset[data_config.test_split],
            batch_size=data_config.batch_size,
            collate_fn=collator,
        ),
    }
