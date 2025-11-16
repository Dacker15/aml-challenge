import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.mixup import mixup_data
from src.training.model import MLP
from src.testing.metrics import mrr


def train_model(
    model: MLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_val_unique: torch.Tensor,
    gt_indices_val: torch.Tensor,
    parameters: dict,
) -> MLP:
    """Train the MLP model with optional mixup and early stopping.
    Args:
        model: MLP model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        y_val_unique: Unique validation targets for MRR calculation
        gt_indices_val: Ground truth indices for validation data
        parameters: Dictionary of training parameters and configurations
    Returns:
        Trained MLP model
    """
    EPOCHS = parameters.get("EPOCHS", 20)
    DEVICE = parameters.get("DEVICE", "cpu")
    LR = parameters.get("LR", 1e-4)
    LABEL_SMOOTHING = parameters.get("LABEL_SMOOTHING", 0.1)
    TEMPERATURE = parameters.get("TEMPERATURE", 0.05)
    WEIGHT_DECAY = parameters.get("WEIGHT_DECAY", 1e-4)
    MIXUP_ALPHA = parameters.get("MIXUP_ALPHA", 0.4)
    ACCUMULATION_STEPS = parameters.get("ACCUMULATION_STEPS", 1)

    MODEL_PATH = parameters.get("MODEL_PATH", "best_model.pth")

    USE_ADAM = parameters.get("USE_ADAM", False)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    if USE_ADAM:
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(EPOCHS * len(train_loader)) // ACCUMULATION_STEPS
    )

    # Initialize metrics tracking
    all_train_loss, all_val_loss = [], []
    best_mrr = float("-inf")
    all_val_mrr = []

    # Early stopping variables
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        for X_batch, y_batch in progress_bar:
            # Move data to selected device
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            X_batch = X_batch.float()
            y_batch = y_batch.float()

            optimizer.zero_grad()

            # Apply mixup to data and targets
            X_batch, y_a, y_b, lam = mixup_data(X_batch, y_batch, alpha=MIXUP_ALPHA)

            # 1. Get translated text (X) embeddings
            translated_X = model(X_batch)

            # 2. Normalize both embeddings
            translated_X = F.normalize(translated_X, p=2, dim=-1)
            y_a = F.normalize(y_a, p=2, dim=-1)
            y_b = F.normalize(y_b, p=2, dim=-1)

            # 3. Compute logits and loss for both targets
            logits_a = (translated_X @ y_a.T) / TEMPERATURE
            logits_b = (translated_X @ y_b.T) / TEMPERATURE
            labels = torch.arange(logits_a.shape[0]).to(DEVICE)
            loss_a = (criterion(logits_a, labels) + criterion(logits_a.T, labels)) / 2.0
            loss_b = (criterion(logits_b, labels) + criterion(logits_b.T, labels)) / 2.0
            # 4. Combine losses using mixup lambda
            loss = lam * loss_a + (1 - lam) * loss_b

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            progress_bar.set_postfix({"Train Loss": loss.item()})

        train_loss /= len(train_loader)
        all_train_loss.append(train_loss)

        progress_bar.set_postfix({"Train Loss": train_loss})

        # Validation
        model.eval()
        val_loss = 0
        epoch_val_predictions = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # Move data to selected device
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                # 1. Get translated text (X) embeddings
                translated_X = model(X_batch)

                # 2. Normalize both embeddings
                translated_X = F.normalize(translated_X, p=2, dim=-1)
                y_batch = F.normalize(y_batch, p=2, dim=-1)

                # 3. Compute logits and loss
                logits = (translated_X @ y_batch.T) / TEMPERATURE
                labels = torch.arange(logits.shape[0]).to(DEVICE)
                loss = (criterion(logits, labels) + criterion(logits.T, labels)) / 2.0
                val_loss += loss.item()
                epoch_val_predictions.append(translated_X.cpu())

            val_loss /= len(val_loader)
            all_val_loss.append(val_loss)

            # Compute MRR over the entire validation set
            epoch_val_predictions = torch.cat(epoch_val_predictions, dim=0)
            epoch_mrr = mrr(epoch_val_predictions, y_val_unique, gt_indices_val)
            all_val_mrr.append(epoch_mrr)

        print(
            f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val MRR={epoch_mrr:.4f}"
        )

        if epoch_mrr > best_mrr:
            # Variable upgrade
            best_mrr = epoch_mrr
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Saved best model (MRR={best_mrr:.4f})")
        elif epochs_without_improvement < 5:
            epochs_without_improvement += 1
        else:
            print(
                f"\n✗ Early stopping: No improvement for 5 epochs. Best MRR={best_mrr:.4f}"
            )
            break

    return model, all_train_loss, all_val_loss, all_val_mrr
