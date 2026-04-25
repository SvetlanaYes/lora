import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from masteringmasters.analysis import analyze_weight_updates
from masteringmasters.config import AnalysisConfig
from masteringmasters.registry import (
    build_experiment_config,
    list_model_keys,
    list_task_keys,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze fine-tuning update matrices using per-layer rank metrics and "
            "truncated-SVD reconstruction errors."
        )
    )
    parser.add_argument("--model-key", choices=list_model_keys(), default="distilbert")
    parser.add_argument("--task-key", choices=list_task_keys(), default="emotion")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    experiment = build_experiment_config(args.model_key, args.task_key)
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    if checkpoint_path is None:
        if args.epoch > 0:
            checkpoint_path = experiment.checkpoint_dir / f"epoch_{args.epoch:02d}.pth"
        else:
            checkpoint_path = experiment.best_model_dir / "best_model.pth"
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else experiment.analysis_dir / "weight_update_ranks.csv"
    )
    summary_csv = experiment.analysis_dir / "layer_type_summary.csv"
    summary_json = experiment.analysis_dir / "analysis_summary.json"
    print("=" * 72)
    print("WEIGHT-UPDATE ANALYSIS")
    print(f"model_key: {args.model_key}")
    print(f"task_key: {args.task_key}")
    print(f"model_name: {experiment.model_name}")
    print(f"checkpoint_path: {checkpoint_path}")
    print(f"output_csv: {output_csv}")
    print(f"summary_csv: {summary_csv}")
    print(f"summary_json: {summary_json}")
    print("=" * 72)
    analysis_config = AnalysisConfig(checkpoint_path=checkpoint_path)
    result = analyze_weight_updates(
        model_name=experiment.model_name,
        num_labels=experiment.num_labels,
        analysis_config=analysis_config,
        output_csv=output_csv,
        summary_csv=summary_csv,
        summary_json=summary_json,
    )
    print(f"Analyzed layers: {len(result['layer_rows'])}")
    print(f"Layer groups summarized: {len(result['summary_rows'])}")
    print(json.dumps(result["summary_rows"], indent=2))
    print(f"Analysis saved to {output_csv}")
    print(f"Layer-type summary saved to {summary_csv}")
    print(f"Analysis metadata saved to {summary_json}")


if __name__ == "__main__":
    main()
