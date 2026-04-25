from pathlib import Path

import torch


def checkpoint_path(checkpoint_dir: Path, epoch: int) -> Path:
    return checkpoint_dir / f"epoch_{epoch:02d}.pth"


def save_checkpoint(model, checkpoint_dir: Path, epoch: int) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(checkpoint_dir, epoch)
    torch.save(model.state_dict(), path)
    return path


def save_named_checkpoint(model, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(model, path: Path, device: torch.device):
    # Loading directly onto MPS can trigger internal copy/alignment failures
    # for some checkpoints, so always restore on CPU first and let callers move
    # the fully loaded model to the target device afterwards.
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model
