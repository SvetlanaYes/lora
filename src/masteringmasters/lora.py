import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from peft import LoraConfig, TaskType, get_peft_model


@dataclass
class LoraExperimentSpec:
    mode: str
    rank: int
    alpha: int
    dropout: float
    target_modules: List[str]
    rank_pattern: Dict[str, int]


def default_lora_targets(model_key: str) -> List[str]:
    if model_key == "distilbert":
        return ["q_lin", "v_lin"]
    if model_key in {"bert", "roberta"}:
        return ["query", "value"]
    raise ValueError(f"Unsupported LoRA model key: {model_key}")


def _matches_any(name: str, targets: Iterable[str]) -> bool:
    return any(part in name for part in targets)


def _snap_rank(value: float, allowed_ranks: List[int]) -> int:
    for rank in allowed_ranks:
        if value <= rank:
            return rank
    return allowed_ranks[-1]


def build_adaptive_rank_pattern(
    model,
    analysis_csv: Path,
    target_modules: List[str],
    min_rank: int = 4,
    max_rank: int = 32,
) -> Dict[str, int]:
    """Map LoRA target modules to ranks using full fine-tuning rank signals.

    The rule is intentionally simple and thesis-friendly:

    - read per-layer `rank_at_90_ratio` from full fine-tuning analysis
    - convert the ratio into a rank between `min_rank` and `max_rank`
    - snap to common LoRA ranks: 4, 8, 16, 32

    This uses full fine-tuning update compressibility as a proxy for how much
    adaptation capacity each LoRA layer should receive.
    """
    if not analysis_csv.exists():
        raise FileNotFoundError(f"Missing analysis CSV for adaptive LoRA: {analysis_csv}")

    allowed_ranks = [rank for rank in [4, 8, 16, 32, 64, 128] if min_rank <= rank <= max_rank]
    if not allowed_ranks:
        raise ValueError("No allowed ranks remain after applying min_rank/max_rank.")

    rank_by_layer = {}
    with analysis_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            layer_name = row["layer_name"]
            if not _matches_any(layer_name, target_modules):
                continue
            rank_ratio = float(row.get("rank_at_90_ratio") or 0.0)
            raw_rank = min_rank + rank_ratio * (max_rank - min_rank)
            rank_by_layer[layer_name.replace(".weight", "")] = _snap_rank(raw_rank, allowed_ranks)

    rank_pattern = {}
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and _matches_any(module_name, target_modules):
            rank_pattern[module_name] = rank_by_layer.get(module_name, min_rank)

    if not rank_pattern:
        raise ValueError(
            f"No LoRA target modules matched targets={target_modules}. "
            "Check model architecture and target module names."
        )

    return rank_pattern


def build_lora_spec(
    model_key: str,
    mode: str,
    rank: int,
    alpha: int,
    dropout: float,
    model=None,
    analysis_csv: Path = None,
    min_rank: int = 4,
    max_rank: int = 32,
) -> LoraExperimentSpec:
    target_modules = default_lora_targets(model_key)
    rank_pattern = {}
    if mode == "adaptive":
        if model is None or analysis_csv is None:
            raise ValueError("Adaptive LoRA requires both a model and an analysis CSV.")
        rank_pattern = build_adaptive_rank_pattern(
            model=model,
            analysis_csv=analysis_csv,
            target_modules=target_modules,
            min_rank=min_rank,
            max_rank=max_rank,
        )

    return LoraExperimentSpec(
        mode=mode,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
        rank_pattern=rank_pattern,
    )


def apply_lora(model, spec: LoraExperimentSpec):
    config_kwargs = {
        "task_type": TaskType.SEQ_CLS,
        "r": spec.rank,
        "lora_alpha": spec.alpha,
        "lora_dropout": spec.dropout,
        "target_modules": spec.target_modules,
        "bias": "none",
    }
    if spec.rank_pattern:
        config_kwargs["rank_pattern"] = spec.rank_pattern
    config = LoraConfig(**config_kwargs)
    return get_peft_model(model, config)


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percent": (100.0 * trainable / total) if total else 0.0,
    }
