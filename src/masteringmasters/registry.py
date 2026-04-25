from dataclasses import dataclass
from typing import Optional

from .config import DataConfig, ExperimentConfig


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_name: str
    num_labels: Optional[int] = None


@dataclass(frozen=True)
class TaskSpec:
    key: str
    dataset_name: str
    num_labels: int
    dataset_config_name: str = ""
    text_field: str = "text"
    label_field: str = "label"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"


MODEL_REGISTRY = {
    "distilbert": ModelSpec(
        key="distilbert",
        model_name="distilbert-base-uncased",
    ),
    "bert": ModelSpec(
        key="bert",
        model_name="bert-base-uncased",
    ),
    "roberta": ModelSpec(
        key="roberta",
        model_name="FacebookAI/roberta-base",
    ),
}


TASK_REGISTRY = {
    "emotion": TaskSpec(
        key="emotion",
        dataset_name="dair-ai/emotion",
        num_labels=6,
    ),
    "clinc_oos": TaskSpec(
        key="clinc_oos",
        dataset_name="clinc/clinc_oos",
        dataset_config_name="plus",
        num_labels=151,
        label_field="intent",
    ),
    "ag_news": TaskSpec(
        key="ag_news",
        dataset_name="ag_news",
        num_labels=4,
    ),
}


def list_model_keys():
    return tuple(MODEL_REGISTRY.keys())


def list_task_keys():
    return tuple(TASK_REGISTRY.keys())


def build_run_name(model_key: str, task_key: str) -> str:
    return f"{model_key}_{task_key}"


def resolve_specs(model_key: str, task_key: str):
    model_spec = MODEL_REGISTRY[model_key]
    task_spec = TASK_REGISTRY[task_key]
    return model_spec, task_spec


def build_experiment_config(model_key: str, task_key: str) -> ExperimentConfig:
    model_spec, task_spec = resolve_specs(model_key, task_key)
    return ExperimentConfig(
        model_name=model_spec.model_name,
        num_labels=task_spec.num_labels,
        run_name=build_run_name(model_key, task_key),
    )


def build_data_config(task_key: str, batch_size: int = 8, max_length: int = 256) -> DataConfig:
    task_spec = TASK_REGISTRY[task_key]
    return DataConfig(
        dataset_name=task_spec.dataset_name,
        dataset_config_name=task_spec.dataset_config_name,
        dataset_text_field=task_spec.text_field,
        dataset_label_field=task_spec.label_field,
        train_split=task_spec.train_split,
        validation_split=task_spec.validation_split,
        test_split=task_spec.test_split,
        batch_size=batch_size,
        max_length=max_length,
    )
