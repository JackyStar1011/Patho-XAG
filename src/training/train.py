import torch
import torch.nn as nn
from torch.optim import AdamW

from src.data.dataloader import create_dataloader
from src.models.resnet import create_resnet50
from src.training.engine import (
    train_one_epoch,
    validate_one_epoch,
)


def main():
    # -------------------------
    # Config
    # -------------------------
    num_epochs = 1
    batch_size = 1
    learning_rate = 1e-4
    num_workers = 0

    smoke_test = True

    if smoke_test: 
        max_train_batches = 1
        max_val_batches = 30
    else:
        max_train_batches = None
        max_val_batches = None

    # -------------------------
    # Device
    # -------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # -------------------------
    # Data
    # -------------------------
    train_loader = create_dataloader(
        split="train",
        mode="whole",
        batch_size=batch_size,
        num_workers=num_workers,
    )

    val_loader = create_dataloader(
        split="val",
        mode="whole",
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # -------------------------
    # Model
    # -------------------------
    model = create_resnet50(
        num_classes=2,
        pretrained=True,
    )

    model = model.to(device)

    # -------------------------
    # Loss
    # -------------------------
    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # Optimizer
    # -------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(1, num_epochs + 1):

        print(
            f"\nEpoch {epoch}/{num_epochs}"
        )

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=max_train_batches,
        )

        val_loss, val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=max_val_batches,
        )

        print(f"Train Loss: {train_loss:.4f}")

        print(f"Val Loss: {val_loss:.4f}")

        for name, value in val_metrics.items():
            print(f"{name}: {value:.4f}")

if __name__ == "__main__":
    main()
