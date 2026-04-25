from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_scheduler

from .checkpoints import save_checkpoint, save_named_checkpoint
from .config import TrainingConfig
from .evaluator import evaluate_model
from .utils import cleanup_device, get_autocast_context


def _evaluate_training_epoch(model, dataloader, device, use_mixed_precision: bool) -> float:
    correct = 0
    total = 0
    model.eval()

    for batch in tqdm(dataloader, desc="Train Accuracy", leave=False):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            with get_autocast_context(device, use_mixed_precision):
                outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == batch["labels"]).sum().item()
        total += batch["labels"].numel()
        cleanup_device(device)

    return (correct / total) if total else 0.0


def _write_metrics_report(history: List[Dict], best_epoch: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "MASTERINGMASTERS Training Report",
        "",
        f"Best validation epoch: {best_epoch}",
        "",
    ]

    for row in history:
        lines.extend(
            [
                f"Epoch {row['epoch']}",
                f"  train_accuracy: {row['train_accuracy']:.6f}",
                f"  validation_accuracy: {row['validation_accuracy']:.6f}",
                f"  test_accuracy: {row['test_accuracy']:.6f}",
                f"  checkpoint: {row['checkpoint']}",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def train(
    model,
    dataloaders,
    device,
    training_config: TrainingConfig,
    checkpoint_dir: Path,
    best_model_dir: Path,
):
    optimizer = AdamW(model.parameters(), lr=training_config.learning_rate)
    total_steps = training_config.num_epochs * len(dataloaders["train"])
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    history = []
    progress = tqdm(range(total_steps), desc="Training")
    model.train()
    best_validation_accuracy = float("-inf")
    best_epoch = -1
    best_checkpoint_path = None

    for epoch in range(1, training_config.num_epochs + 1):
        print(f"\nStarting epoch {epoch}/{training_config.num_epochs}")
        for batch in dataloaders["train"]:
            batch = {key: value.to(device) for key, value in batch.items()}
            with get_autocast_context(device, training_config.use_mixed_precision):
                outputs = model(**batch)
                loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.update(1)
            cleanup_device(device)

        checkpoint = None
        if training_config.checkpoint_every_epoch:
            checkpoint = save_checkpoint(model, checkpoint_dir, epoch)
            print(f"Saved checkpoint: {checkpoint}")

        train_accuracy = _evaluate_training_epoch(
            model,
            dataloaders["train"],
            device,
            training_config.use_mixed_precision,
        )
        validation_metrics = evaluate_model(
            model,
            dataloaders["validation"],
            device,
            training_config.use_mixed_precision,
        )
        test_metrics = evaluate_model(
            model,
            dataloaders["test"],
            device,
            training_config.use_mixed_precision,
        )
        validation_accuracy = validation_metrics["accuracy"]
        test_accuracy = test_metrics["accuracy"]

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            best_checkpoint_path = save_named_checkpoint(
                model,
                best_model_dir,
                "best_model.pth",
            )
            print(
                f"New best validation checkpoint at epoch {epoch}: "
                f"val_accuracy={validation_accuracy:.6f} path={best_checkpoint_path}"
            )

        history.append(
            {
                "epoch": epoch,
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
                "test_accuracy": test_accuracy,
                "checkpoint": str(checkpoint) if checkpoint else None,
            }
        )
        print(
            f"Epoch {epoch} complete | "
            f"train={train_accuracy:.6f} "
            f"val={validation_accuracy:.6f} "
            f"test={test_accuracy:.6f}"
        )
        model.train()

    metrics_report_path = best_model_dir / "metrics_report.txt"
    _write_metrics_report(history, best_epoch, metrics_report_path)
    print(f"Metrics report saved to {metrics_report_path}")

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else None,
        "metrics_report_path": str(metrics_report_path),
    }
