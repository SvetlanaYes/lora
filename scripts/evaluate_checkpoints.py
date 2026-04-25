import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from masteringmasters.checkpoints import checkpoint_path, load_checkpoint
from masteringmasters.config import DataConfig, ExperimentConfig, TrainingConfig
from masteringmasters.data import build_dataloaders, tokenize_dataset
from masteringmasters.evaluator import evaluate_model
from masteringmasters.models import load_sequence_classifier, load_tokenizer
from masteringmasters.registry import (
    build_data_config,
    build_experiment_config,
    list_model_keys,
    list_task_keys,
)
from masteringmasters.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoints.")
    parser.add_argument("--model-key", choices=list_model_keys(), default="distilbert")
    parser.add_argument("--task-key", choices=list_task_keys(), default="emotion")
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    experiment = build_experiment_config(args.model_key, args.task_key)
    data_config = build_data_config(args.task_key)
    training_config = TrainingConfig(num_epochs=args.num_epochs)

    print("=" * 72)
    print("CHECKPOINT EVALUATION")
    print(f"model_key: {args.model_key}")
    print(f"task_key: {args.task_key}")
    print(f"model_name: {experiment.model_name}")
    print(f"dataset_name: {data_config.dataset_name}")
    if data_config.dataset_config_name:
        print(f"dataset_config_name: {data_config.dataset_config_name}")
    print(f"checkpoint_dir: {experiment.checkpoint_dir}")
    print(f"split: {args.split}")
    print("=" * 72)

    tokenizer = load_tokenizer(experiment.model_name)
    tokenized_dataset = tokenize_dataset(tokenizer, data_config)
    dataloaders = build_dataloaders(tokenized_dataset, tokenizer, data_config)
    device = get_device()
    print(f"Device: {device}")

    results = []
    for epoch in range(1, training_config.num_epochs + 1):
        model = load_sequence_classifier(experiment.model_name, experiment.num_labels)
        path = checkpoint_path(experiment.checkpoint_dir, epoch)
        if not path.exists():
            print(f"Skipping missing checkpoint: {path}")
            continue
        print(f"Evaluating checkpoint: {path}")
        model = load_checkpoint(model, path, device)
        model.to(device)
        metrics = evaluate_model(
            model,
            dataloaders[args.split],
            device,
            training_config.use_mixed_precision,
        )
        results.append({"epoch": epoch, "checkpoint": str(path), "metrics": metrics})
        print(f"Epoch {epoch} metrics: {metrics}")

    print(
        json.dumps(
            {
                "model_key": args.model_key,
                "task_key": args.task_key,
                "split": args.split,
                "results": results,
            },
            indent=2,
        )
    )
    print("Checkpoint evaluation finished.")


if __name__ == "__main__":
    main()
