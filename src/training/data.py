import numpy as np
import pandas as pd
import torch

from collections import defaultdict
from pathlib import Path


def load_data(path: Path):
    """Load processed data from .npz file"""
    return dict(np.load(path, allow_pickle=True))


def prepare_train_data(data: dict):
    """Prepare training data from loaded dict"""
    caption_embd = data["captions/embeddings"]
    image_embd = data["images/embeddings"]
    # Map caption embeddings to corresponding image embeddings
    label = data["captions/label"]  # N x M

    print(f"Train data: {len(caption_embd)} captions, {len(image_embd)} images")

    # repeat the image embeddings according to the label
    label_idx = np.nonzero(label)[1]
    image_embd = image_embd[label_idx]
    assert caption_embd.shape[0] == image_embd.shape[0], (
        "Mismatch in number of caption and image embeddings"
    )

    X = torch.from_numpy(caption_embd).float()
    # Map each caption to its corresponding image embedding
    y = torch.from_numpy(image_embd).float()

    return X, y


def create_image_based_split(
    captions_file_path: Path, total_samples: int, train_ratio=0.9, random_seed=42
):
    """Create a train/val split based on images, ensuring all captions for an image are in the same set."""
    df = pd.read_csv(captions_file_path)
    image_to_indices = defaultdict(list)
    for idx, row in df.iterrows():
        image_to_indices[row["image"]].append(idx)

    unique_images = list(image_to_indices.keys())
    rng = np.random.default_rng(random_seed)
    rng.shuffle(unique_images)

    n_train_images = int(len(unique_images) * train_ratio)
    train_images = set(unique_images[:n_train_images])

    train_mask = torch.zeros(total_samples, dtype=torch.bool)
    for idx, row in df.iterrows():
        if row["image"] in train_images:
            train_mask[idx] = True

    val_mask = ~train_mask
    print(f"Split created: Train={train_mask.sum()}, Val={val_mask.sum()}")
    return train_mask, val_mask


def load_coco_and_align(coco_path: Path):
    """Load COCO dataset and align captions with corresponding image embeddings."""
    print(f"Loading external dataset: {coco_path}...")
    try:
        data = np.load(coco_path)
        txt_emb = data["captions/embeddings"]
        img_indices = data["captions/image_indices"]
        img_emb_all = data["images/embeddings"]
        target_img_emb = img_emb_all[img_indices]

        print(f"COCO Loaded: {len(txt_emb)} samples.")
        return torch.from_numpy(txt_emb), torch.from_numpy(target_img_emb)
    except FileNotFoundError:
        print("COCO file not found! Training will proceed with challenge data only.")
        return None, None
