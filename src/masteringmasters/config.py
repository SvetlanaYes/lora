from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    dataset_name: str = "emotion"
    dataset_config_name: str = ""
    dataset_text_field: str = "text"
    dataset_label_field: str = "label"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    max_length: int = 256
    batch_size: int = 8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    split_seed: int = 42


@dataclass
class TrainingConfig:
    learning_rate: float = 5e-5
    num_epochs: int = 8
    checkpoint_every_epoch: bool = True
    use_mixed_precision: bool = True


@dataclass
class AnalysisConfig:
    checkpoint_path: Path
    energy_thresholds: tuple[float, ...] = (0.9, 0.95, 0.99)
    exclude_embeddings: bool = True
    reconstruction_ranks: tuple[int, ...] = (4, 8, 16, 32, 64)


@dataclass
class ExperimentConfig:
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 6
    output_root: Path = field(default_factory=lambda: Path("artifacts"))
    run_name: str = "distilbert_emotion_baseline"

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_name

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def best_model_dir(self) -> Path:
        return self.run_dir / "best_model"

    @property
    def analysis_dir(self) -> Path:
        return self.run_dir / "analysis"
