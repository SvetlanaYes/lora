import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from masteringmasters.registry import list_model_keys, list_task_keys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect full fine-tuning training summaries into thesis-ready CSV tables."
    )
    parser.add_argument("--artifacts-root", default=str(ROOT / "artifacts"))
    parser.add_argument(
        "--output-csv",
        default=str(ROOT / "outputs" / "tables" / "full_finetuning_results.csv"),
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Only include DistilBERT and BERT rows, leaving RoBERTa as optional.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    artifacts_root = Path(args.artifacts_root)
    model_keys = [key for key in list_model_keys() if not args.core_only or key in {"distilbert", "bert"}]
    rows = []

    for model_key in model_keys:
        for task_key in list_task_keys():
            run_name = f"{model_key}_{task_key}"
            summary_path = artifacts_root / run_name / "training_summary.json"
            if not summary_path.exists():
                print(f"Skipping missing summary: {summary_path}")
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "run_name": run_name,
                    "model_key": model_key,
                    "task_key": task_key,
                    "model_name": summary.get("model_name"),
                    "dataset_name": summary.get("dataset_name"),
                    "best_epoch": summary.get("best_epoch"),
                    "best_validation_accuracy": summary.get("best_validation_accuracy"),
                    "final_test_accuracy": summary.get("test_metrics", {}).get("accuracy"),
                    "best_checkpoint_path": summary.get("best_checkpoint_path"),
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

    print(f"Collected {len(rows)} full fine-tuning result rows.")
    print(f"Saved to {output_csv}")


if __name__ == "__main__":
    main()
