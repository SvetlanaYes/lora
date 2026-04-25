import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from masteringmasters.registry import list_model_keys, list_task_keys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate per-run rank-analysis CSV files across experiment runs."
    )
    parser.add_argument(
        "--artifacts-root",
        default=str(ROOT / "artifacts"),
        help="Root directory containing <model>_<task> run folders.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(ROOT / "outputs" / "analysis" / "all_runs_layer_metrics.csv"),
        help="Combined per-layer analysis CSV output path.",
    )
    parser.add_argument(
        "--output-summary-csv",
        default=str(ROOT / "outputs" / "analysis" / "all_runs_group_summary.csv"),
        help="Combined per-run grouped summary CSV output path.",
    )
    return parser.parse_args()


def _read_csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main():
    args = parse_args()
    artifacts_root = Path(args.artifacts_root)
    combined_layer_rows = []
    combined_summary_rows = []

    for model_key in list_model_keys():
        for task_key in list_task_keys():
            run_name = f"{model_key}_{task_key}"
            run_dir = artifacts_root / run_name
            layer_csv = run_dir / "analysis" / "weight_update_ranks.csv"
            summary_csv = run_dir / "analysis" / "layer_type_summary.csv"
            if not layer_csv.exists():
                print(f"Skipping missing analysis file: {layer_csv}")
                continue

            print(f"Aggregating run: {run_name}")
            for row in _read_csv_rows(layer_csv):
                row["run_name"] = run_name
                row["model_key"] = model_key
                row["task_key"] = task_key
                combined_layer_rows.append(row)

            if summary_csv.exists():
                for row in _read_csv_rows(summary_csv):
                    row["run_name"] = run_name
                    row["model_key"] = model_key
                    row["task_key"] = task_key
                    combined_summary_rows.append(row)

    _write_csv(Path(args.output_csv), combined_layer_rows)
    if combined_summary_rows:
        _write_csv(Path(args.output_summary_csv), combined_summary_rows)

    print(f"Combined layer metrics saved to {args.output_csv}")
    if combined_summary_rows:
        print(f"Combined grouped summaries saved to {args.output_summary_csv}")


if __name__ == "__main__":
    main()
