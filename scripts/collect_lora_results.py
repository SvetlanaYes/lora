import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect LoRA training summaries into thesis-ready CSV tables."
    )
    parser.add_argument("--artifacts-root", default=str(ROOT / "artifacts_lora"))
    parser.add_argument(
        "--output-csv",
        default=str(ROOT / "outputs" / "tables" / "lora_results.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    artifacts_root = Path(args.artifacts_root)
    rows = []

    if not artifacts_root.exists():
        print(f"LoRA artifacts root does not exist yet: {artifacts_root}")
    else:
        for summary_path in sorted(artifacts_root.glob("*/lora_training_summary.json")):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            params = summary.get("parameter_counts", {})
            rows.append(
                {
                    "run_name": summary_path.parent.name,
                    "model_key": summary.get("model_key"),
                    "task_key": summary.get("task_key"),
                    "method": f"lora_{summary.get('lora_mode')}",
                    "lora_rank": summary.get("lora_rank"),
                    "lora_min_rank": summary.get("lora_min_rank"),
                    "lora_max_rank": summary.get("lora_max_rank"),
                    "best_epoch": summary.get("best_epoch"),
                    "best_validation_accuracy": summary.get("best_validation_accuracy"),
                    "final_test_accuracy": summary.get("test_metrics", {}).get("accuracy"),
                    "trainable_parameters": params.get("trainable_parameters"),
                    "total_parameters": params.get("total_parameters"),
                    "trainable_percent": params.get("trainable_percent"),
                    "adapter_dir": summary.get("adapter_dir"),
                    "metrics_report_path": summary.get("metrics_report_path"),
                }
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"Collected {len(rows)} LoRA result rows.")
    print(f"Saved to {output_csv}")


if __name__ == "__main__":
    main()
