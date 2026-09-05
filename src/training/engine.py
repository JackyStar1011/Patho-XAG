import torch
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    max_batches=None,
):
    """
    Train model for one epoch.

    Returns:
        average_loss: float
    """

    model.train()

    running_loss = 0.0
    num_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False,
    )

    for batch_idx, batch in enumerate(progress_bar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Reset gradients from previous step
        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # Compute loss
        loss = criterion(
            logits,
            labels,
        )

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        num_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = running_loss / num_samples

    return average_loss


@torch.no_grad()
def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device,
    max_batches=None
):
    """
    Evaluate model on validation/test data.

    Returns:
        average_loss: float
        metrics: dict
    """

    model.eval()

    running_loss = 0.0
    num_samples = 0

    all_labels = []
    all_probabilities = []

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        leave=False,
    )

    for batch_idx, batch in enumerate(progress_bar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        logits = model(images)

        # Loss
        loss = criterion(
            logits,
            labels,
        )

        # Convert logits -> probability of Adenoma
        probabilities = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        num_samples += batch_size

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

    average_loss = running_loss / num_samples

    metrics = compute_metrics(
        labels=all_labels,
        probabilities=all_probabilities,
    )

    return average_loss, metrics
