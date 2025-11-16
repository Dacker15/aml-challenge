import numpy as np
import torch
import torch.nn.functional as F

from datetime import datetime
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from src.common.device import get_device
from src.testing.metrics import mrr
from src.testing.submission import generate_submission
from src.training.data import load_data, prepare_train_data, create_image_based_split
from src.training.ensemble import Ensemble
from src.training.model import MLP
from src.training.train import train_model

EXPERIMENT_DATETIME = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_DIR = f"experiments/experiment_{EXPERIMENT_DATETIME}"
EXPERIMENT_INFO_PATH = f"{EXPERIMENT_DIR}/info.txt"
EXPERIMENT_SUBMISSION_PATH = f"{EXPERIMENT_DIR}/submission.csv"
MODEL_PATH = f"{EXPERIMENT_DIR}/mlp.pth"
ENSEMBLE_PATH = f"{EXPERIMENT_DIR}/ensemble.pth"


def main():
    BATCH_SIZE = 512
    ACCUMULATION_STEPS = 1
    VIRTUAL_BATCH_SIZE = BATCH_SIZE * ACCUMULATION_STEPS
    DEVICE = get_device()
    RANDOM_SEED = 42

    config_dict = {
        "EXPERIMENT_DATETIME": EXPERIMENT_DATETIME,
        "EPOCHS": 20,
        "BATCH_SIZE": BATCH_SIZE,
        "LR": 0.0001,
        "DROPOUT": 0.3,
        "WEIGHT_DECAY": 1e-5,
        "LABEL_SMOOTHING": 0.1,
        "TEMPERATURE": 0.02,
        "NUM_LAYERS": 3,
        "INPUT_DIM": 1024,
        "HIDDEN_DIM": 1024,
        "OUTPUT_DIM": 1536,
        "ACCUMULATION_STEPS": ACCUMULATION_STEPS,
        "VIRTUAL_BATCH_SIZE": VIRTUAL_BATCH_SIZE,
        "DEVICE": str(DEVICE),
        "MODEL_PATH": MODEL_PATH,
        "USE_ADAM": True,
    }

    emsemble_variations = [
        {"NUM_LAYERS": 3, "DROPOUT": 0.5, "LR": 0.0001, "TEMPERATURE": 0.020},
        {"NUM_LAYERS": 3, "DROPOUT": 0.48, "LR": 0.00009, "TEMPERATURE": 0.018},
        {"NUM_LAYERS": 4, "DROPOUT": 0.5, "LR": 0.0001, "TEMPERATURE": 0.020},
        {"NUM_LAYERS": 3, "DROPOUT": 0.52, "LR": 0.00011, "TEMPERATURE": 0.022},
        {"NUM_LAYERS": 3, "DROPOUT": 0.5, "LR": 0.0001, "TEMPERATURE": 0.019},
    ]

    print("Experiment configuration:")
    for key, value in config_dict.items():
        print(f"{key}: {value}")

    print("\nVariations for Ensemble:")
    for idx, variation in enumerate(emsemble_variations, start=1):
        print(f"Variation {idx}:")
        for key, value in variation.items():
            print(f"{key}: {value}")
        print("-----")

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

    # Load and prepare data
    print("\nLoading and preparing data...")
    train_data = load_data(train_data_path)
    X_data, y_data = prepare_train_data(train_data)

    models = []
    performances = []
    histories = []

    for idx, variation in enumerate(emsemble_variations, start=1):
        print(f"\nTraining model {idx} for Ensemble")
        # Update config with variation
        for key, value in variation.items():
            config_dict[key] = value
        config_dict["MODEL_PATH"] = f"{EXPERIMENT_DIR}/mlp_model_{idx}.pth"

        train_mask, val_mask = create_image_based_split(
            train_captions_path, len(X_data)
        )
        X_train, y_train = X_data[train_mask], y_data[train_mask]
        X_val, y_val = X_data[val_mask], y_data[val_mask]

        # Bagging at 80% of training data
        num_train_samples = X_train.shape[0]
        bag_size = int(0.8 * num_train_samples)
        rng = np.random.default_rng(RANDOM_SEED + idx)
        bag_indices = rng.choice(num_train_samples, size=bag_size, replace=False)

        X_train_bag = X_train[bag_indices]
        y_train_bag = y_train[bag_indices]

        print(f"Bagging: Using {len(X_train_bag)}/{len(X_train)} samples")

        # Create DataLoaders
        train_dataset = TensorDataset(X_train_bag, y_train_bag)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Get unique validation targets and inverse indices for evaluation
        y_val_unique, inverse_indices = torch.unique(y_val, dim=0, return_inverse=True)

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

        model.load_state_dict(torch.load(config_dict["MODEL_PATH"]))

        models.append(model)
        histories.append(
            {
                "train_loss": all_train_loss,
                "val_loss": all_val_loss,
                "val_mrr": all_val_mrr,
            }
        )
        performances.append(all_val_mrr.index(max(all_val_mrr)))  # Epoch with best MRR

    print("\nEnsemble training completed.")
    print(f"  Mean:   {np.mean(performances):.6f}")
    print(f"  Std:    {np.std(performances):.6f}")
    print(f"  Min:    {np.min(performances):.6f}")
    print(f"  Max:    {np.max(performances):.6f}")

    # Save ensemble
    ensemble_data = {
        "model_states": [model.state_dict() for model in models],
        "performances": performances,
        "n_models": len(models),
        "timestamp": datetime.now().isoformat(),
    }
    torch.save(ensemble_data, ENSEMBLE_PATH)
    print(f"\n✓ Ensemble saved to: {ENSEMBLE_PATH}")

    ensemble = Ensemble(models=models)
    train_mask, val_mask = create_image_based_split(
        train_captions_path,
        len(X_data),
    )
    ensemble_pred = ensemble.predict(X_val, device=DEVICE)

    # Normalize
    ensemble_pred_normalized = F.normalize(ensemble_pred, p=2, dim=-1)
    y_val_normalized = F.normalize(y_val, p=2, dim=-1)
    targets = torch.arange(len(y_val))
    ensemble_mrr = mrr(ensemble_pred_normalized, y_val_normalized, targets)

    mean_individual = np.mean(performances)
    improvement = ensemble_mrr - mean_individual
    print(f"Mean individual MRR: {mean_individual:.6f}")
    print(f"Ensemble MRR: {ensemble_mrr:.6f}")
    print(
        f"Improvement: +{improvement:.6f} ({improvement / mean_individual * 100:.2f}%)"
    )

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
