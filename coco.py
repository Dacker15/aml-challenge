import gc
import torch

from datetime import datetime
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from src.common.device import get_device
from src.common.plots import get_training_plot
from src.testing.submission import generate_submission
from src.training.data import (
    load_data,
    prepare_train_data,
    create_image_based_split,
    load_coco_and_align,
)
from src.training.model import MLP
from src.training.train import train_model

EXPERIMENT_DATETIME = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_DIR = f"experiments/experiment_{EXPERIMENT_DATETIME}"
EXPERIMENT_INFO_PATH = f"{EXPERIMENT_DIR}/info.txt"
EXPERIMENT_SUBMISSION_PATH = f"{EXPERIMENT_DIR}/submission.csv"
MODEL_PATH = f"{EXPERIMENT_DIR}/mlp.pth"
TRAINING_PLOT_PATH = f"{EXPERIMENT_DIR}/training_plots.png"


def main():
    BATCH_SIZE = 1024
    ACCUMULATION_STEPS = 1
    VIRTUAL_BATCH_SIZE = BATCH_SIZE * ACCUMULATION_STEPS
    DEVICE = get_device()
    ORIGINAL_DATASET_WEIGHT = 10.0

    config_dict = {
        "EXPERIMENT_DATETIME": EXPERIMENT_DATETIME,
        "EPOCHS": 25,
        "BATCH_SIZE": BATCH_SIZE,
        "LR": 0.0001,
        "DROPOUT": 0.3,
        "WEIGHT_DECAY": 1e-4,
        "LABEL_SMOOTHING": 0.1,
        "TEMPERATURE": 0.02,
        "NUM_LAYERS": 3,
        "INPUT_DIM": 1024,
        "HIDDEN_DIM": 1024,
        "OUTPUT_DIM": 1536,
        "ACCUMULATION_STEPS": ACCUMULATION_STEPS,
        "VIRTUAL_BATCH_SIZE": VIRTUAL_BATCH_SIZE,
        "DEVICE": str(DEVICE),
        "ORIGINAL_DATASET_WEIGHT": 10.0,
        "MIXUP_ALPHA": 0.2,
        "MODEL_PATH": MODEL_PATH,
        "TRAINING_PLOT_PATH": TRAINING_PLOT_PATH,
    }
    print("Experiment configuration:")
    for key, value in config_dict.items():
        print(f"{key}: {value}")

    # Create experiment directory if it doesn't exist
    Path(EXPERIMENT_DIR).mkdir(parents=True, exist_ok=True)

    # Save config to file
    with open(EXPERIMENT_INFO_PATH, "w") as f:
        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")
    print(f"\nExperiment configuration saved to: {EXPERIMENT_INFO_PATH}")

    # Dataset paths
    train_data_path = Path("data/train/train.npz")
    train_captions_path = Path("data/train/captions.txt")
    test_data_path = Path("data/test/test.clean.npz")
    coco_path = Path("data/coco2017/coco_embeddings_clean.npz")

    # Load and prepare data
    print("\nLoading and preparing data...")
    train_data = load_data(train_data_path)
    X_data, y_data = prepare_train_data(train_data)
    train_mask, val_mask = create_image_based_split(train_captions_path, len(X_data))
    X_train, y_train = X_data[train_mask], y_data[train_mask]
    X_val, y_val = X_data[val_mask], y_data[val_mask]
    print(
        f"Training data: {len(X_train)} samples, Validation data: {len(X_val)} samples"
    )
    # Load and align COCO data
    print("\nLoading and aligning COCO data...")
    X_coco, y_coco = load_coco_and_align(coco_path)

    # Merge COCO data with training data
    num_original_train_samples = len(X_train)
    X_train_merged = torch.cat([X_train, X_coco], dim=0)
    y_train_merged = torch.cat([y_train, y_coco], dim=0)

    print(
        f"Merged training data: {len(X_train_merged)} samples ({len(X_coco)} COCO samples added)"
    )

    # Clean up
    del X_data, y_data, X_coco, y_coco, train_mask, val_mask
    gc.collect()

    # Get unique validation targets and inverse indices for evaluation
    y_val_unique, inverse_indices = torch.unique(y_val, dim=0, return_inverse=True)

    # Apply Stratified Sampling to training data and prepare training data loader
    print("\nPreparing data loaders with stratified sampling...")
    train_dataset = TensorDataset(X_train_merged, y_train_merged)
    weights = torch.zeros(len(X_train_merged))
    weights[:num_original_train_samples] = ORIGINAL_DATASET_WEIGHT
    weights[num_original_train_samples:] = 1.0
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=len(X_train_merged), replacement=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        pin_memory=True,
        num_workers=2,
    )

    # Prepare validation data loader
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )

    print(
        f"✓ Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}"
    )

    # Initialize the model
    model = MLP(
        input_dim=config_dict["INPUT_DIM"],
        hidden_dim=config_dict["HIDDEN_DIM"],
        output_dim=config_dict["OUTPUT_DIM"],
        num_layers=config_dict["NUM_LAYERS"],
        dropout=config_dict["DROPOUT"],
    ).to(DEVICE)

    # Start training
    model, all_train_loss, all_val_loss, all_val_mrr = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        y_val_unique=y_val_unique,
        gt_indices_val=inverse_indices,
        parameters=config_dict,
    )

    get_training_plot(
        all_train_loss,
        all_val_loss,
        all_val_mrr,
        TRAINING_PLOT_PATH,
    )

    # Save the trained model
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Generate submission on test set
    test_data = load_data(test_data_path)
    test_embds = torch.from_numpy(test_data["captions/embeddings"]).float().to(DEVICE)

    with torch.no_grad():
        pred_embds = model(test_embds).cpu()

    generate_submission(
        test_data["captions/ids"], pred_embds, EXPERIMENT_SUBMISSION_PATH
    )

    print(f"✓ Submission saved: {EXPERIMENT_SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
