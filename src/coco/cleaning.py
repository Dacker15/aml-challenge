import numpy as np
from pathlib import Path
from typing import Set, List, Tuple
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def load_captions_from_txt(txt_path: Path, logger: logging.Logger = None) -> List[str]:
    """
    Loads captions from a .txt file (format: id,caption).
    Returns only the captions (without id).
    """
    if logger:
        logger.info(f"Loading captions from {txt_path}")
    captions = []

    with open(txt_path, "r", encoding="utf-8") as f:
        # Skip the first line (header)
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on first comma (id,caption)
            parts = line.split(",", 1)
            if len(parts) == 2:
                captions.append(parts[1])

    logger.info(f"Loaded {len(captions)} captions to remove")
    return captions


def encode_reference_captions(
    text_model: SentenceTransformer,
    captions: List[str],
    batch_size: int = 256,
    logger: logging.Logger = None,
) -> np.ndarray:
    """
    Encode reference captions using the same model as preprocessing.
    """
    if logger:
        logger.info(f"Encoding {len(captions)} reference captions")
    embeddings = text_model.encode(
        captions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    if logger:
        logger.info(f"Reference embeddings shape: {embeddings.shape}")
    return embeddings


def find_similar_indices(
    dataset_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    threshold: float = 0.88,
    batch_size: int = 1000,
    logger: logging.Logger = None,
) -> Set[int]:
    """
    Finds the indices of captions in the dataset that are too similar
    to the reference captions (using cosine similarity).

    Returns:
        Set of indices to remove
    """
    logger.info(f"Searching for similar captions (threshold={threshold})")
    indices_to_remove = set()

    n_dataset = dataset_embeddings.shape[0]

    # Process in batches to avoid out of memory
    for start_idx in tqdm(range(0, n_dataset, batch_size), desc="Embeddings comparison"):
        end_idx = min(start_idx + batch_size, n_dataset)
        batch_embeddings = dataset_embeddings[start_idx:end_idx]

        # Compute cosine similarity between batch and all references
        similarities = cosine_similarity(batch_embeddings, reference_embeddings)

        # For each caption in the batch, check if it is similar to any reference
        max_similarities = similarities.max(axis=1)

        # Find local indices that exceed the threshold
        local_indices = np.where(max_similarities >= threshold)[0]

        # Convert to global indices
        global_indices = local_indices + start_idx
        indices_to_remove.update(global_indices.tolist())

    logger.info(f"Found {len(indices_to_remove)} similar captions")
    return indices_to_remove


def expand_to_full_images(
    caption_indices_to_remove: Set[int],
    caption_image_indices: np.ndarray,
    logger: logging.Logger = None,
) -> Tuple[Set[int], Set[int]]:
    """
    Expands removal: when a caption is identified as similar,
    removes ALL captions of the same image.

    Args:
        caption_indices_to_remove: Indices of similar captions found
        caption_image_indices: Array mapping caption -> image_idx

    Returns:
        (expanded_caption_indices, image_indices_to_remove)
    """
    logger.info("Expanding removal to all captions of the involved images")

    # Find the indices of images to remove
    images_to_remove = set()
    for caption_idx in caption_indices_to_remove:
        image_idx = caption_image_indices[caption_idx]
        images_to_remove.add(image_idx)

    logger.info(f"Images to remove: {len(images_to_remove)}")

    # Find ALL captions associated with these images
    expanded_caption_indices = set()
    for caption_idx, image_idx in enumerate(caption_image_indices):
        if image_idx in images_to_remove:
            expanded_caption_indices.add(caption_idx)

    logger.info(
        f"Total captions to remove (after expansion): {len(expanded_caption_indices)}"
    )

    return expanded_caption_indices, images_to_remove


def save_removed_images_list(
    output_path: Path, removed_image_names: List[str], logger: logging.Logger = None
):
    """
    Saves the list of removed images to a text file.
    """
    logger.info(f"Saving list of removed images: {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Images removed from the dataset\n")
        f.write(f"# Total: {len(removed_image_names)}\n")
        f.write("#" + "=" * 50 + "\n\n")

        for img_name in sorted(removed_image_names):
            f.write(f"{img_name}\n")

    logger.info(f"✓ Saved {len(removed_image_names)} removed images")


def filter_npz_file(
    input_npz: Path,
    output_npz: Path,
    captions_txt: Path,
    removed_images_txt: Path,
    threshold: float = 0.88,
    text_model_name: str = "sentence-transformers/roberta-large-nli-stsb-mean-tokens",
    logger: logging.Logger = None,
):
    """
    Filters the .npz file by removing similar captions and images.
    """
    # 1) Load the text model (same used to create the dataset)
    logger.info(f"Loading text model: {text_model_name}")
    text_model = SentenceTransformer(text_model_name)

    # 2) Load captions to remove
    reference_captions = load_captions_from_txt(captions_txt, logger=logger)

    # 3) Encode reference captions
    reference_embeddings = encode_reference_captions(
        text_model, reference_captions, logger=logger
    )

    # 4) Load the .npz file
    logger.info(f"Loading .npz file: {input_npz}")
    data = np.load(str(input_npz), allow_pickle=True)

    caption_embeddings = data["captions/embeddings"]
    caption_image_indices = data["captions/image_indices"]
    image_embeddings = data["images/embeddings"]
    image_names = data["images/names"]

    logger.info(
        f"Original dataset: {caption_embeddings.shape[0]} captions, {image_embeddings.shape[0]} images"
    )

    # 5) Find indices of similar captions
    similar_caption_indices = find_similar_indices(
        caption_embeddings, reference_embeddings, threshold=threshold, logger=logger
    )

    # 6) Expand removal to all captions of the same images
    caption_indices_to_remove, image_indices_to_remove = expand_to_full_images(
        similar_caption_indices, caption_image_indices, logger=logger
    )

    # 7) Save the list of removed images
    removed_image_names = [image_names[idx] for idx in sorted(image_indices_to_remove)]
    save_removed_images_list(removed_images_txt, removed_image_names, logger=logger)

    # 8) Create mask to keep only valid captions
    keep_caption_mask = np.ones(caption_embeddings.shape[0], dtype=bool)
    keep_caption_mask[list(caption_indices_to_remove)] = False

    # 9) Filter captions
    filtered_caption_embeddings = caption_embeddings[keep_caption_mask]
    filtered_caption_image_indices = caption_image_indices[keep_caption_mask]

    logger.info(f"Remaining captions: {filtered_caption_embeddings.shape[0]}")

    # 10) Create mask for images to keep
    keep_image_mask = np.ones(image_embeddings.shape[0], dtype=bool)
    keep_image_mask[list(image_indices_to_remove)] = False

    # 11) Create mapping old indices -> new indices for images
    old_to_new_image_idx = {}
    new_image_idx = 0

    for old_idx in range(image_embeddings.shape[0]):
        if keep_image_mask[old_idx]:
            old_to_new_image_idx[old_idx] = new_image_idx
            new_image_idx += 1

    # 12) Filter images
    filtered_image_embeddings = image_embeddings[keep_image_mask]
    filtered_image_names = image_names[keep_image_mask]

    logger.info(f"Remaining images: {filtered_image_embeddings.shape[0]}")

    # 13) Update image indices in captions
    updated_caption_image_indices = np.array(
        [old_to_new_image_idx[old_idx] for old_idx in filtered_caption_image_indices],
        dtype=np.int32,
    )

    # 14) Prepare the new .npz file
    meta = {
        "metadata/num_captions": np.array(
            [filtered_caption_embeddings.shape[0]], dtype=np.int64
        ),
        "metadata/num_images": np.array(
            [filtered_image_embeddings.shape[0]], dtype=np.int64
        ),
        "metadata/embedding_dim_text": np.array(
            [filtered_caption_embeddings.shape[1]], dtype=np.int64
        ),
        "metadata/embedding_dim_image": np.array(
            [filtered_image_embeddings.shape[1]], dtype=np.int64
        ),
    }

    filtered_data = {
        **meta,
        "captions/embeddings": filtered_caption_embeddings,
        "captions/image_indices": updated_caption_image_indices,
        "images/names": filtered_image_names,
        "images/embeddings": filtered_image_embeddings,
    }

    # 15) Save the filtered file
    logger.info(f"Saving filtered file: {output_npz}")
    np.savez_compressed(str(output_npz), **filtered_data)

    logger.info("=" * 60)
    logger.info("✓ FILTERING COMPLETED!")
    logger.info("=" * 60)
    logger.info("Statistics:")
    logger.info(f"  - Similar captions found: {len(similar_caption_indices)}")
    logger.info(
        f"  - Images identified for removal: {len(image_indices_to_remove)}"
    )
    logger.info(
        f"  - Total captions removed: {len(caption_indices_to_remove)} ({len(caption_indices_to_remove) / caption_embeddings.shape[0] * 100:.2f}%)"
    )
    logger.info(
        f"  - Images removed: {len(image_indices_to_remove)} ({len(image_indices_to_remove) / image_embeddings.shape[0] * 100:.2f}%)"
    )
    logger.info("")
    logger.info("Final dataset:")
    logger.info(
        f"  - Captions: {filtered_caption_embeddings.shape[0]} (from {caption_embeddings.shape[0]})"
    )
    logger.info(
        f"  - Images: {filtered_image_embeddings.shape[0]} (from {image_embeddings.shape[0]})"
    )
    logger.info("")
    logger.info("Generated files:")
    logger.info(f"  - Filtered dataset: {output_npz}")
    logger.info(f"  - List of removed images: {removed_images_txt}")
    logger.info("=" * 60)
