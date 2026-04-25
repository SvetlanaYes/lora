import evaluate
import torch
from tqdm.auto import tqdm

from .utils import cleanup_device, get_autocast_context


def evaluate_model(model, dataloader, device: torch.device, use_mixed_precision: bool):
    metric = evaluate.load("accuracy")
    model.eval()

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            with get_autocast_context(device, use_mixed_precision):
                outputs = model(**batch)

        predictions = torch.argmax(outputs.logits, dim=-1)
        metric.add_batch(
            predictions=predictions.cpu(),
            references=batch["labels"].cpu(),
        )
        cleanup_device(device)

    return metric.compute()
