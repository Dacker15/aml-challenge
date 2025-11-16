import matplotlib.pyplot as plt

from pathlib import Path


def get_training_plot(
    train_loss: list,
    val_loss: list,
    val_mrr: list,
    training_plot_path: Path,
):
    """
    Plots training and validation metrics and loss curves and highlights the epoch with minimum value for each.
    Args:
        train_loss (list): List of training losses per epoch.
        val_loss (list): List of validation losses per epoch.
        val_mrr (list): List of validation MRR-GT based per epoch.
        training_plot_path (Path): Path to save the training plot.
    """
    # Fake epoch numbers for x-axis
    epochs = list(range(1, len(train_loss) + 1))

    # Find the epoch with the minimum validation loss
    min_val_loss_epoch_idx = val_loss.index(min(val_loss))
    min_val_loss = val_loss[min_val_loss_epoch_idx]
    min_val_loss_epoch = min_val_loss_epoch_idx + 1

    # Find the epoch with the best validation MRR-GT based
    best_val_mrr_idx = val_mrr.index(max(val_mrr))
    best_val_mrr = val_mrr[best_val_mrr_idx]
    best_val_mrr_epoch = best_val_mrr_idx + 1

    rows = 1
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(20, 12))

    # Loss plot
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train Loss", marker="o")
    ax.plot(epochs, val_loss, label="Validation Loss", marker="x")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True)

    ax.scatter(
        min_val_loss_epoch,
        min_val_loss,
        color="red",
        s=100,
        zorder=5,
        label=f"Min Val Loss - {min_val_loss:.4f} (Epoch {min_val_loss_epoch})",
    )
    ax.legend()

    # MRR-GT based plot
    ax = axes[1]
    ax.plot(epochs, val_mrr, label="Validation MRR-GT based", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MRR-GT based")
    ax.set_title("Validation MRR-GT based")
    ax.legend()
    ax.grid(True)

    ax.scatter(
        best_val_mrr_epoch,
        best_val_mrr,
        color="red",
        s=100,
        zorder=5,
        label=f"Best Val MRR-GT based - {best_val_mrr:.4f} (Epoch {best_val_mrr_epoch})",
    )
    ax.legend()

    plt.savefig(training_plot_path)
    plt.show()
