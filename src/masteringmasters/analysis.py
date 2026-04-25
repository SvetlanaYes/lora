import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .checkpoints import load_checkpoint
from .config import AnalysisConfig
from .models import load_sequence_classifier


def _layer_type(layer_name: str) -> str:
    lowered = layer_name.lower()
    if "embed" in lowered:
        return "embeddings"
    if "classifier" in lowered or "pre_classifier" in lowered or "score" in lowered:
        return "classifier"
    if "attention" in lowered or ".attn." in lowered:
        return "attention"
    if "intermediate" in lowered or ".mlp." in lowered or ".ffn." in lowered:
        return "ffn"
    if ".output.dense" in lowered and "attention" not in lowered:
        return "ffn"
    return "other"


def _layer_depth(layer_name: str) -> int:
    parts = layer_name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part == "layer":
            try:
                return int(parts[index + 1])
            except ValueError:
                return -1
    return -1


def _is_weight_matrix(name: str, tensor: torch.Tensor, exclude_embeddings: bool) -> bool:
    if not name.endswith(".weight"):
        return False
    if exclude_embeddings and "embed" in name:
        return False
    return tensor.ndim >= 2


def _reshape_for_svd(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.float()
    return tensor.view(tensor.shape[0], -1).float()


def _effective_rank(singular_values: torch.Tensor) -> float:
    total = singular_values.sum()
    if total <= 0:
        return 0.0
    probabilities = singular_values / total
    entropy = -(probabilities * torch.log(probabilities + 1e-12)).sum()
    return float(torch.exp(entropy).item())


def _participation_ratio(singular_values: torch.Tensor) -> float:
    if singular_values.numel() == 0:
        return 0.0
    energy = singular_values.pow(2)
    denominator = energy.pow(2).sum()
    if denominator <= 0:
        return 0.0
    numerator = energy.sum().pow(2)
    return float((numerator / denominator).item())


def _stable_rank(matrix: torch.Tensor, singular_values: torch.Tensor) -> float:
    if singular_values.numel() == 0 or singular_values[0] <= 0:
        return 0.0
    fro_norm_sq = torch.linalg.norm(matrix, ord="fro").pow(2)
    spectral_sq = singular_values[0].pow(2)
    return float((fro_norm_sq / spectral_sq).item())


def _energy_rank(singular_values: torch.Tensor, threshold: float) -> int:
    if singular_values.numel() == 0:
        return 0
    energy = singular_values.pow(2)
    cumulative = torch.cumsum(energy, dim=0)
    total = energy.sum()
    if total <= 0:
        return 0
    return int(torch.searchsorted(cumulative / total, threshold).item() + 1)


def _relative_reconstruction_error(singular_values: torch.Tensor, rank: int) -> float:
    if singular_values.numel() == 0:
        return 0.0
    energy = singular_values.pow(2)
    total_energy = energy.sum()
    if total_energy <= 0:
        return 0.0
    kept_energy = energy[:rank].sum()
    residual_energy = torch.clamp(total_energy - kept_energy, min=0.0)
    return float(torch.sqrt(residual_energy / total_energy).item())


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _build_group_summaries(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["layer_type"]].append(row)

    summary_rows = []
    for layer_type, layer_rows in sorted(grouped.items()):
        summary = {
            "layer_type": layer_type,
            "num_layers": len(layer_rows),
            "mean_full_rank": round(_mean([row["full_rank"] for row in layer_rows]), 4),
            "mean_effective_rank": round(_mean([row["effective_rank"] for row in layer_rows]), 4),
            "mean_effective_rank_ratio": round(
                _mean([row["effective_rank_ratio"] for row in layer_rows]),
                4,
            ),
            "mean_stable_rank": round(_mean([row["stable_rank"] for row in layer_rows]), 4),
            "mean_participation_ratio": round(
                _mean([row["participation_ratio"] for row in layer_rows]),
                4,
            ),
            "mean_rank_at_90": round(_mean([row["rank_at_90"] for row in layer_rows]), 4),
            "mean_rank_at_95": round(_mean([row["rank_at_95"] for row in layer_rows]), 4),
            "mean_rank_at_99": round(_mean([row["rank_at_99"] for row in layer_rows]), 4),
            "mean_frobenius_norm": round(_mean([row["frobenius_norm"] for row in layer_rows]), 6),
        }
        for key in sorted(layer_rows[0].keys()):
            if key.startswith("relative_error_r"):
                summary[f"mean_{key}"] = round(_mean([row[key] for row in layer_rows]), 6)
        summary_rows.append(summary)

    return summary_rows


def _write_csv(output_path: Path, rows: List[Dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def _write_json(output_path: Path, payload: Dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def analyze_weight_updates(
    model_name: str,
    num_labels: int,
    analysis_config: AnalysisConfig,
    output_csv: Optional[Path] = None,
    summary_csv: Optional[Path] = None,
    summary_json: Optional[Path] = None,
):
    """Analyze fine-tuning update matrices layer by layer.

    The analysis uses the update matrix ΔW = W_finetuned - W_pretrained and
    computes several complementary notions of low-rank structure:

    - energy-based ranks at multiple thresholds
    - stable rank
    - entropy-based effective rank
    - participation ratio
    - relative reconstruction error for fixed truncated-SVD ranks

    These metrics are intended for empirical comparison across layers, tasks,
    and models before designing adaptive LoRA rules.
    """
    pretrained = load_sequence_classifier(model_name, num_labels)
    fine_tuned = load_sequence_classifier(model_name, num_labels)
    device = torch.device("cpu")
    fine_tuned = load_checkpoint(fine_tuned, analysis_config.checkpoint_path, device)

    rows = []
    for layer_name, pretrained_weights in pretrained.state_dict().items():
        if not _is_weight_matrix(
            layer_name,
            pretrained_weights,
            analysis_config.exclude_embeddings,
        ):
            continue

        fine_tuned_weights = fine_tuned.state_dict()[layer_name]
        delta = _reshape_for_svd((fine_tuned_weights - pretrained_weights).cpu())
        if delta.numel() == 0:
            continue

        singular_values = torch.linalg.svdvals(delta)
        full_rank = min(delta.shape)
        effective_rank = _effective_rank(singular_values)
        stable_rank = _stable_rank(delta, singular_values)
        participation_ratio = _participation_ratio(singular_values)
        frobenius_norm = float(torch.linalg.norm(delta, ord="fro").item())
        row = {
            "layer_name": layer_name,
            "layer_type": _layer_type(layer_name),
            "layer_depth": _layer_depth(layer_name),
            "rows": delta.shape[0],
            "cols": delta.shape[1],
            "full_rank": full_rank,
            "effective_rank": round(effective_rank, 4),
            "effective_rank_ratio": round(effective_rank / full_rank, 4) if full_rank else 0.0,
            "stable_rank": round(stable_rank, 4),
            "stable_rank_ratio": round(stable_rank / full_rank, 4) if full_rank else 0.0,
            "participation_ratio": round(participation_ratio, 4),
            "participation_ratio_ratio": round(participation_ratio / full_rank, 4) if full_rank else 0.0,
            "frobenius_norm": round(frobenius_norm, 6),
            "top_singular_value": round(float(singular_values[0].item()) if singular_values.numel() else 0.0, 6),
        }
        for threshold in analysis_config.energy_thresholds:
            key = f"rank_at_{int(threshold * 100)}"
            energy_rank = _energy_rank(singular_values, threshold)
            row[key] = energy_rank
            row[f"{key}_ratio"] = round(energy_rank / full_rank, 4) if full_rank else 0.0
        for rank in analysis_config.reconstruction_ranks:
            safe_rank = min(rank, full_rank)
            row[f"relative_error_r{rank}"] = round(
                _relative_reconstruction_error(singular_values, safe_rank),
                6,
            )
        rows.append(row)

    summary_rows = _build_group_summaries(rows)

    if output_csv is not None:
        _write_csv(output_csv, rows)
    if summary_csv is not None:
        _write_csv(summary_csv, summary_rows)
    if summary_json is not None:
        payload = {
            "checkpoint_path": str(analysis_config.checkpoint_path),
            "num_layers": len(rows),
            "layer_type_summary": summary_rows,
        }
        _write_json(summary_json, payload)

    return {
        "layer_rows": rows,
        "summary_rows": summary_rows,
    }
