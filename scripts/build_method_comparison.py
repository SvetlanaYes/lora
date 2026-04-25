import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine full fine-tuning and LoRA result tables into one comparison CSV."
    )
    parser.add_argument(
        "--full-ft-csv",
        default=str(ROOT / "outputs" / "tables" / "full_finetuning_results.csv"),
    )
    parser.add_argument(
        "--lora-csv",
        default=str(ROOT / "outputs" / "tables" / "lora_results.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(ROOT / "outputs" / "tables" / "method_comparison.csv"),
    )
    return parser.parse_args()


def _read_rows(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    args = parse_args()
    rows = []

    for row in _read_rows(Path(args.full_ft_csv)):
        rows.append(
            {
                "run_name": row["run_name"],
                "model_key": row["model_key"],
                "task_key": row["task_key"],
                "method": "full_finetuning",
                "rank": "",
                "best_epoch": row["best_epoch"],
                "best_validation_accuracy": row["best_validation_accuracy"],
                "final_test_accuracy": row["final_test_accuracy"],
                "trainable_parameters": "",
                "trainable_percent": "",
                "artifact_path": row["best_checkpoint_path"],
            }
        )

    for row in _read_rows(Path(args.lora_csv)):
        rows.append(
            {
                "run_name": row["run_name"],
                "model_key": row["model_key"],
                "task_key": row["task_key"],
                "method": row["method"],
                "rank": row["lora_rank"],
                "best_epoch": row["best_epoch"],
                "best_validation_accuracy": row["best_validation_accuracy"],
                "final_test_accuracy": row["final_test_accuracy"],
                "trainable_parameters": row["trainable_parameters"],
                "trainable_percent": row["trainable_percent"],
                "artifact_path": row["adapter_dir"],
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "run_name",
            "model_key",
            "task_key",
            "method",
            "rank",
            "best_epoch",
            "best_validation_accuracy",
            "final_test_accuracy",
            "trainable_parameters",
            "trainable_percent",
            "artifact_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Combined {len(rows)} method rows.")
    print(f"Saved to {output_csv}")


if __name__ == "__main__":
    main()
