import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
from masteringmasters.trainer import train
from masteringmasters.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train a sequence classifier.")
    parser.add_argument("--model-key", choices=list_model_keys(), default="distilbert")
    parser.add_argument("--task-key", choices=list_task_keys(), default="emotion")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-epochs", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    experiment = build_experiment_config(args.model_key, args.task_key)
    data_config = build_data_config(
        args.task_key,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
    )

    print("=" * 72)
    print("TRAINING RUN")
    print(f"model_key: {args.model_key}")
    print(f"task_key: {args.task_key}")
    print(f"model_name: {experiment.model_name}")
    print(f"dataset_name: {data_config.dataset_name}")
    if data_config.dataset_config_name:
        print(f"dataset_config_name: {data_config.dataset_config_name}")
    print(f"run_dir: {experiment.run_dir}")
    print(f"checkpoint_dir: {experiment.checkpoint_dir}")
    print(f"best_model_dir: {experiment.best_model_dir}")
    print(f"num_epochs: {training_config.num_epochs}")
    print("=" * 72)

    tokenizer = load_tokenizer(experiment.model_name)
    print("Tokenizer loaded.")
    tokenized_dataset = tokenize_dataset(tokenizer, data_config)
    print("Dataset loaded and tokenized.")
    dataloaders = build_dataloaders(tokenized_dataset, tokenizer, data_config)
    print("Dataloaders created.")

    device = get_device()
    print(f"Device: {device}")
    model = load_sequence_classifier(experiment.model_name, experiment.num_labels)
    model.to(device)
    print("Model loaded.")

    training_result = train(
        model,
        dataloaders,
        device,
        training_config,
        experiment.checkpoint_dir,
        experiment.best_model_dir,
    )
    print("Training phase finished.")
    test_metrics = evaluate_model(
        model,
        dataloaders["test"],
        device,
        training_config.use_mixed_precision,
    )
    print("Final test evaluation finished.")

    experiment.run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = experiment.run_dir / "training_summary.json"
    summary = {
        "model_key": args.model_key,
        "task_key": args.task_key,
        "model_name": experiment.model_name,
        "dataset_name": data_config.dataset_name,
        "history": training_result["history"],
        "best_epoch": training_result["best_epoch"],
        "best_validation_accuracy": training_result["best_validation_accuracy"],
        "best_checkpoint_path": training_result["best_checkpoint_path"],
        "metrics_report_path": training_result["metrics_report_path"],
        "test_metrics": test_metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best epoch: {training_result['best_epoch']}")
    print(f"Best checkpoint: {training_result['best_checkpoint_path']}")
    print(f"Metrics report: {training_result['metrics_report_path']}")
    print(json.dumps(summary, indent=2))
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
