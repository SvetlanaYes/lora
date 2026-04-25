import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from masteringmasters.config import TrainingConfig
from masteringmasters.data import build_dataloaders, tokenize_dataset
from masteringmasters.evaluator import evaluate_model
from masteringmasters.lora import apply_lora, build_lora_spec, count_parameters
from masteringmasters.models import load_sequence_classifier, load_tokenizer
from masteringmasters.registry import build_data_config, build_experiment_config, list_task_keys
from masteringmasters.trainer import train
from masteringmasters.utils import get_device


CORE_MODEL_KEYS = ("distilbert", "bert")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train fixed-rank or adaptive-rank LoRA sequence-classification models."
    )
    parser.add_argument("--model-key", choices=CORE_MODEL_KEYS, default="distilbert")
    parser.add_argument("--task-key", choices=list_task_keys(), default="emotion")
    parser.add_argument("--mode", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--min-rank", type=int, default=4)
    parser.add_argument("--max-rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "artifacts_lora"),
        help="Root directory for LoRA experiment outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the LoRA model and print parameter counts without training.",
    )
    return parser.parse_args()


def _lora_run_name(model_key: str, task_key: str, mode: str, rank: int, min_rank: int, max_rank: int):
    if mode == "fixed":
        return f"{model_key}_{task_key}_fixed_r{rank}"
    return f"{model_key}_{task_key}_adaptive_r{min_rank}_{max_rank}"


def main():
    args = parse_args()
    base_experiment = build_experiment_config(args.model_key, args.task_key)
    lora_run_name = _lora_run_name(
        args.model_key,
        args.task_key,
        args.mode,
        args.rank,
        args.min_rank,
        args.max_rank,
    )
    run_dir = Path(args.output_root) / lora_run_name
    checkpoint_dir = run_dir / "checkpoints"
    best_model_dir = run_dir / "best_model"
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
    print("LORA TRAINING RUN")
    print(f"model_key: {args.model_key}")
    print(f"task_key: {args.task_key}")
    print(f"mode: {args.mode}")
    print(f"rank: {args.rank}")
    print(f"run_dir: {run_dir}")
    print("=" * 72)

    model = load_sequence_classifier(base_experiment.model_name, base_experiment.num_labels)

    analysis_csv = base_experiment.analysis_dir / "weight_update_ranks.csv"
    lora_spec = build_lora_spec(
        model_key=args.model_key,
        mode=args.mode,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        model=model,
        analysis_csv=analysis_csv,
        min_rank=args.min_rank,
        max_rank=args.max_rank,
    )
    model = apply_lora(model, lora_spec)
    param_counts = count_parameters(model)
    print(f"LoRA targets: {lora_spec.target_modules}")
    if lora_spec.rank_pattern:
        print(f"Adaptive rank pattern entries: {len(lora_spec.rank_pattern)}")
    print(f"Trainable parameters: {param_counts['trainable_parameters']}")
    print(f"Trainable percent: {param_counts['trainable_percent']:.4f}%")
    model.print_trainable_parameters()

    if args.dry_run:
        print("Dry run complete. No training was executed.")
        return

    tokenizer = load_tokenizer(base_experiment.model_name)
    tokenized_dataset = tokenize_dataset(tokenizer, data_config)
    dataloaders = build_dataloaders(tokenized_dataset, tokenizer, data_config)

    device = get_device()
    model.to(device)
    training_result = train(
        model=model,
        dataloaders=dataloaders,
        device=device,
        training_config=training_config,
        checkpoint_dir=checkpoint_dir,
        best_model_dir=best_model_dir,
    )
    test_metrics = evaluate_model(
        model,
        dataloaders["test"],
        device,
        training_config.use_mixed_precision,
    )

    adapter_dir = best_model_dir / "adapter"
    model.save_pretrained(adapter_dir)
    summary = {
        "model_key": args.model_key,
        "task_key": args.task_key,
        "base_model_name": base_experiment.model_name,
        "dataset_name": data_config.dataset_name,
        "lora_mode": args.mode,
        "lora_rank": args.rank,
        "lora_alpha": args.alpha,
        "lora_dropout": args.dropout,
        "lora_min_rank": args.min_rank,
        "lora_max_rank": args.max_rank,
        "target_modules": lora_spec.target_modules,
        "rank_pattern": lora_spec.rank_pattern,
        "parameter_counts": param_counts,
        "history": training_result["history"],
        "best_epoch": training_result["best_epoch"],
        "best_validation_accuracy": training_result["best_validation_accuracy"],
        "best_checkpoint_path": training_result["best_checkpoint_path"],
        "adapter_dir": str(adapter_dir),
        "metrics_report_path": training_result["metrics_report_path"],
        "test_metrics": test_metrics,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "lora_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"LoRA adapter saved to {adapter_dir}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
