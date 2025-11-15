import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm

from torch.utils.data import DataLoader

from mixup import mixup_data
from model import MLP
from testing.metrics import mrr, mrr2, ndcg, recall_at_k


def train_model(
    model: MLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_val_unique,
    gt_indices_val,
    parameters: dict,
    show_plot=True,
):
    """Train the MLP model with optional mixup and early stopping.
    Args:
        model: MLP model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        y_val_unique: Unique validation targets for MRR calculation
        gt_indices_val: Ground truth indices for validation data
        parameters: Dictionary of training parameters and configurations
        show_plot: Whether to show training/validation loss plot
    Returns:
        Trained MLP model
    """
    EPOCHS = parameters.get("epochs", 20)
    DEVICE = parameters.get("device", "cpu")
    LR = parameters.get("lr", 1e-4)
    LABEL_SMOOTHING = parameters.get("label_smoothing", 0.1)
    TEMPERATURE = parameters.get("temperature", 0.05)
    WEIGHT_DECAY = parameters.get("weight_decay", 1e-4)
    USE_MIXUP = parameters.get("use_mixup", False)
    MIXUP_ALPHA = parameters.get("mixup_alpha", 0.4)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(train_loader)
    )

    # Initialize metrics tracking
    all_train_loss, all_val_loss = [], []
    best_mrr = float("-inf")
    all_val_predictions, all_val_mrr = [], []
    metric_names = [
        "mrr",
        "ndcg",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "recall_at_50",
    ]
    train_metrics = {name: [] for name in metric_names}
    val_metrics = {name: [] for name in metric_names}

    # Early stopping variables
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        # Epoch-level metric accumulators for training
        epoch_train_metrics = {name: [] for name in metric_names}

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        for X_batch, y_batch in progress_bar:
            # Move data to selected device
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            batch_size = X_batch.size(0)

            optimizer.zero_grad()

            if USE_MIXUP:
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
                loss_a = (
                    criterion(logits_a, labels) + criterion(logits_a.T, labels)
                ) / 2.0
                loss_b = (
                    criterion(logits_b, labels) + criterion(logits_b.T, labels)
                ) / 2.0
                # 4. Combine losses using mixup lambda
                loss = lam * loss_a + (1 - lam) * loss_b
            else:
                # 1. Get translated text (X) embeddings
                translated_X = model(X_batch)

                # 2. Normalize both embeddings
                translated_X = F.normalize(translated_X, p=2, dim=-1)
                y_batch = F.normalize(y_batch, p=2, dim=-1)

                # 3. Compute logits and loss
                logits = (translated_X @ y_batch.T) / TEMPERATURE
                labels = torch.arange(logits.shape[0]).to(DEVICE)
                loss = (criterion(logits, labels) + criterion(logits.T, labels)) / 2.0

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

            # Compute metrics for this batch
            with torch.no_grad():
                # Get predicted indices (top-k retrieval)
                if USE_MIXUP:
                    # Calcola similarity per entrambi i target mixup
                    similarity_a = translated_X @ y_a.T
                    similarity_b = translated_X @ y_b.T
                    # Combina le similarity con lo stesso peso lambda
                    similarity = lam * similarity_a + (1 - lam) * similarity_b
                else:
                    similarity = translated_X @ y_batch.T

                pred_indices = (
                    torch.argsort(similarity, dim=1, descending=True).cpu().numpy()
                )
                gt_indices = np.arange(batch_size)

                # Compute each metric
                epoch_train_metrics["mrr"].append(mrr(pred_indices, gt_indices))
                epoch_train_metrics["ndcg"].append(ndcg(pred_indices, gt_indices))
                epoch_train_metrics["recall_at_1"].append(
                    recall_at_k(pred_indices, gt_indices, 1)
                )
                epoch_train_metrics["recall_at_3"].append(
                    recall_at_k(pred_indices, gt_indices, 3)
                )
                epoch_train_metrics["recall_at_5"].append(
                    recall_at_k(pred_indices, gt_indices, 5)
                )
                epoch_train_metrics["recall_at_10"].append(
                    recall_at_k(pred_indices, gt_indices, 10)
                )
                epoch_train_metrics["recall_at_50"].append(
                    recall_at_k(pred_indices, gt_indices, 50)
                )

        train_loss /= len(train_loader)
        all_train_loss.append(train_loss)

        # Average metrics across all batches for this epoch
        for name in metric_names:
            train_metrics[name].append(np.mean(epoch_train_metrics[name]))

        # Validation
        model.eval()
        val_loss = 0

        # Epoch-level metric accumulators for validation
        epoch_val_metrics = {name: [] for name in metric_names}

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

                # Compute metrics for this batch
                similarity = translated_X @ y_batch.T
                pred_indices = (
                    torch.argsort(similarity, dim=1, descending=True).cpu().numpy()
                )
                gt_indices = np.arange(batch_size)

                # Compute each metric
                epoch_val_metrics["mrr"].append(mrr(pred_indices, gt_indices))
                epoch_val_metrics["ndcg"].append(ndcg(pred_indices, gt_indices))
                epoch_val_metrics["recall_at_1"].append(
                    recall_at_k(pred_indices, gt_indices, 1)
                )
                epoch_val_metrics["recall_at_3"].append(
                    recall_at_k(pred_indices, gt_indices, 3)
                )
                epoch_val_metrics["recall_at_5"].append(
                    recall_at_k(pred_indices, gt_indices, 5)
                )
                epoch_val_metrics["recall_at_10"].append(
                    recall_at_k(pred_indices, gt_indices, 10)
                )
                epoch_val_metrics["recall_at_50"].append(
                    recall_at_k(pred_indices, gt_indices, 50)
                )

            val_loss /= len(val_loader)
            all_val_predictions.append(translated_X)
            all_val_loss.append(val_loss)

            # Average metrics across all batches for this epoch
            for name in metric_names:
                val_metrics[name].append(np.mean(epoch_val_metrics[name]))

            # Compute MRR over the entire validation set
            val_predictions_full = torch.cat(all_val_predictions, dim=0)
            epoch_mrr = mrr2(val_predictions_full, y_val_unique, gt_indices_val)
            all_val_mrr.append(epoch_mrr)

        print(
            f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val MRR={epoch_mrr:.4f}"
        )

        if epoch_mrr > best_mrr:
            # Variable upgrade
            best_mrr = epoch_mrr
            epochs_without_improvement = 0
            # torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Saved best model (MRR={best_mrr:.4f})")
        elif epochs_without_improvement < 5:
            epochs_without_improvement += 1
        else:
            print(
                f"\n✗ Early stopping: No improvement for 5 epochs. Best MRR={best_mrr:.4f}"
            )
            break

    # if show_plot:
    #     show_training_plot(all_train_loss, all_val_loss, all_val_mrr)

    return model
