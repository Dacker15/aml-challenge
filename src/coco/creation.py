import json
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from PIL import Image, UnidentifiedImageError
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_text_model(
    model_name: str = "sentence-transformers/roberta-large-nli-stsb-mean-tokens",
    logger: logging.Logger = None,
):
    if logger:
        logger.info(f"Loading text model: {model_name}")
    text_model = SentenceTransformer(model_name)
    return text_model


def load_image_model(
    model_name: str = "facebook/dinov2-giant", logger: logging.Logger = None
):
    if logger:
        logger.info(f"Loading image model: {model_name}")
    image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    image_model = AutoModel.from_pretrained(model_name)
    return image_processor, image_model


def parse_coco_annotations(
    captions_json_path: Path,
    logger: logging.Logger = None,
) -> Tuple[List[str], List[int], Dict[int, str]]:
    """
    Parse COCO captions JSON and return:
    - captions_texts: list of caption strings (ordered by annotation entry)
    - captions_image_ids: list of image_id corresponding to each caption (same order)
    - image_id_to_filename: dict mapping image_id -> file_name (if not present, builds zero-padded)
    """
    if logger:
        logger.info(f"Parsing COCO annotations from: {captions_json_path}")
    with open(captions_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build image_id -> file_name map
    image_id_to_filename = {}
    images = data.get("images", [])
    for img in images:
        img_id = img.get("id")
        file_name = img.get("file_name")
        if file_name is None:
            # fallback: construct zero-padded filename (COCO uses 12 digits)
            file_name = f"{int(img_id):012d}.jpg"
        image_id_to_filename[int(img_id)] = file_name

    # Parse annotations
    captions_texts = []
    captions_image_ids = []
    annotations = data.get("annotations", [])
    for ann in annotations:
        img_id = int(ann.get("image_id"))
        caption = ann.get("caption", "")
        # sometimes caption may be empty - still include, user can filter later
        captions_texts.append(str(caption))
        captions_image_ids.append(img_id)

    logger.info(
        f"Found {len(image_id_to_filename)} image entries and {len(captions_texts)} annotations"
    )
    return captions_texts, captions_image_ids, image_id_to_filename


@torch.inference_mode()
def encode_images_batched(
    image_processor,
    image_model,
    image_filenames: List[str],
    images_root: Path,
    device: str = "cuda",
    batch_size: int = 64,
    use_autocast: bool = True,
    logger: logging.Logger = None,
) -> Tuple[List[str], np.ndarray, List[int]]:
    """
    Batch-encode images. Returns:
    - processed_filenames: list of filenames that were successfully encoded (order matches embeddings)
    - embeddings: np.ndarray shape (N_processed, D)
    - failed_indices: list of indices in the input image_filenames that failed
    """
    if logger:
        logger.info(
            f"Encoding {len(image_filenames)} unique images in batches (batch_size={batch_size})"
        )
    image_model.to(device)
    image_model.eval()

    processed_filenames: List[str] = []
    embeddings_list: List[np.ndarray] = []
    failed_indices: List[int] = []

    # Choose autocast dtype (if GPU)
    use_autocast = use_autocast and ("cuda" in device and torch.cuda.is_available())

    for start in tqdm(
        range(0, len(image_filenames), batch_size), desc="Images batches"
    ):
        batch_files = image_filenames[start : start + batch_size]
        pil_images = []
        valid_file_indices = []  # Original indices in image_filenames that are valid
        for idx_in_batch, fname in enumerate(batch_files):
            img_path = images_root / fname
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    pil_images.append(im.copy())  # Copy to avoid closed file issues
                    valid_file_indices.append(start + idx_in_batch)
            except FileNotFoundError:
                logger.warning(f"Image not found, skipping: {img_path}")
                failed_indices.append(start + idx_in_batch)
            except UnidentifiedImageError:
                logger.warning(
                    f"Cannot identify image file (corrupted?), skipping: {img_path}"
                )
                failed_indices.append(start + idx_in_batch)
            except Exception as e:
                logger.warning(f"Error opening image {img_path}: {e}")
                failed_indices.append(start + idx_in_batch)

        if len(pil_images) == 0:
            continue

        try:
            inputs = image_processor(images=pil_images, return_tensors="pt").to(device)
            # Use autocast for faster FP16 on GPU if supported
            if use_autocast:
                with torch.amp.autocast("cuda"):
                    outputs = image_model(**inputs)
            else:
                outputs = image_model(**inputs)
        except RuntimeError as e:
            logger.error(f"RuntimeError while encoding batch starting at {start}: {e}")
            # Attempt to encode images one-by-one to skip the bad image(s)
            for k, fname in enumerate(batch_files):
                single_path = images_root / fname
                try:
                    with Image.open(single_path) as im:
                        im = im.convert("RGB")
                        single_inputs = image_processor(
                            images=[im], return_tensors="pt"
                        ).to(device)
                        if use_autocast:
                            with torch.cuda.amp.autocast():
                                out = image_model(**single_inputs)
                        else:
                            out = image_model(**single_inputs)
                        if hasattr(out, "last_hidden_state"):
                            feat = out.last_hidden_state.mean(dim=1).cpu().numpy()
                        elif hasattr(out, "pooler_output"):
                            feat = out.pooler_output.cpu().numpy()
                        else:
                            # Fallback: try first tensor value
                            tensor_vals = [
                                v for v in out.values() if torch.is_tensor(v)
                            ]
                            if len(tensor_vals) > 0:
                                feat = tensor_vals[0].mean(dim=1).cpu().numpy()
                            else:
                                raise RuntimeError(
                                    "Cannot extract features from model output"
                                )
                        processed_filenames.append(fname)
                        embeddings_list.append(feat)
                except Exception as e2:
                    logger.warning(f"Failed single image encoding {single_path}: {e2}")
                    failed_indices.append(start + k)
            continue

        # Extract embeddings robustly
        try:
            if (
                hasattr(outputs, "last_hidden_state")
                and outputs.last_hidden_state is not None
            ):
                # Average pool across sequence dimension
                image_feats = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            elif (
                hasattr(outputs, "pooler_output") and outputs.pooler_output is not None
            ):
                image_feats = outputs.pooler_output.cpu().numpy()
            else:
                # Look for any tensor and average across sequence dimension if present
                tensor_vals = [v for v in outputs.values() if torch.is_tensor(v)]
                if len(tensor_vals) > 0:
                    t = tensor_vals[0]
                    if t.dim() == 3:
                        image_feats = t.mean(dim=1).cpu().numpy()
                    elif t.dim() == 2:
                        image_feats = t.cpu().numpy()
                    else:
                        raise RuntimeError(
                            "Unexpected output tensor shape for image model"
                        )
                else:
                    raise RuntimeError(
                        "Cannot extract features from image model outputs"
                    )
        except Exception as e:
            logger.error(
                f"Error extracting features for batch starting at {start}: {e}"
            )
            # mark whole batch as failed
            for k in range(len(batch_files)):
                failed_indices.append(start + k)
            continue

        # Append features and filenames for the valid files only (the processor returns a tensor for valid_images)
        # valid_file_indices aligns with rows in image_feats
        for rel_idx, orig_idx in enumerate(valid_file_indices):
            processed_filenames.append(image_filenames[orig_idx])
            embeddings_list.append(image_feats[rel_idx : rel_idx + 1])  # keep as 2D

    if len(embeddings_list) == 0:
        logger.warning("No image embeddings were produced.")
        return [], np.zeros((0, 0), dtype=np.float32), failed_indices

    embeddings = np.vstack([e for e in embeddings_list])
    logger.info(
        f"Encoded {len(processed_filenames)} images successfully, failed: {len(failed_indices)}"
    )
    return processed_filenames, embeddings, failed_indices


def encode_captions(
    text_model: SentenceTransformer,
    captions: List[str],
    device: str = "cuda",
    batch_size: int = 256,
    logger: logging.Logger = None,
) -> np.ndarray:
    """
    Encode captions using SentenceTransformer.encode. Returns numpy array (N, D).
    """
    if logger:
        logger.info(
            f"Encoding {len(captions)} captions with text model (batch_size={batch_size})"
        )
    # SentenceTransformer will handle batching internally by batch_size argument.
    # Convert device string to something sentence-transformers accepts ('cuda' or 'cpu').
    device_for_st = device
    if device_for_st is None:
        device_for_st = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = text_model.encode(
        captions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device_for_st,
    )
    logger.info(f"Caption embeddings shape: {embeddings.shape}")
    return embeddings


def create_npz_from_coco(
    images_root: Path,
    captions_json: Path,
    output_file: Path,
    device: str = None,
    batch_size_image: int = 64,
    batch_size_text: int = 256,
    logger: logging.Logger = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 1) Parse COCO annotations
    captions_texts, captions_image_ids, image_id_to_filename = parse_coco_annotations(
        captions_json, logger=logger
    )

    # Build unique list of image filenames referenced by annotations (preserve order)
    referenced_filenames = []
    seen = set()
    for img_id in captions_image_ids:
        fname = image_id_to_filename.get(int(img_id))
        if fname is None:
            # Fallback to padded filename if mapping missing
            fname = f"{int(img_id):012d}.jpg"
        if fname not in seen:
            seen.add(fname)
            referenced_filenames.append(fname)

    logger.info(
        f"{len(referenced_filenames)} unique image filenames referenced by captions will be processed (if present)"
    )

    # 2) Load models
    text_model = load_text_model(logger=logger)
    image_processor, image_model = load_image_model(logger=logger)

    # 3) Encode images in batches
    processed_filenames, image_embeddings, failed_image_indices = encode_images_batched(
        image_processor=image_processor,
        image_model=image_model,
        image_filenames=referenced_filenames,
        images_root=images_root,
        device=device,
        batch_size=batch_size_image,
        use_autocast=True,
        logger=logger,
    )

    # If no images encoded, abort
    if len(processed_filenames) == 0:
        logger.error("No images were successfully encoded. Exiting.")
        raise RuntimeError("No images encoded")

    # Map filename -> index into processed_filenames
    images_dict = {fname: idx for idx, fname in enumerate(processed_filenames)}
    logger.info(f"Images dict built: {len(images_dict)} items")

    # 4) Build caption -> image index map, filter captions whose images were not processed
    caption_image_indices = []
    kept_caption_texts = []
    kept_caption_original_indices = []  # optional: index in original captions arrays
    missing_image_count = 0
    for idx, (caption, img_id) in enumerate(zip(captions_texts, captions_image_ids)):
        # Determine filename for this image id
        fname = image_id_to_filename.get(int(img_id))
        if fname is None:
            fname = f"{int(img_id):012d}.jpg"
        img_idx = images_dict.get(fname, None)
        if img_idx is None:
            # Image referenced but was not processed (missing or corrupted)
            missing_image_count += 1
            continue
        # Keep this caption
        kept_caption_texts.append(caption)
        caption_image_indices.append(img_idx)
        kept_caption_original_indices.append(idx)

    logger.info(
        f"Kept {len(kept_caption_texts)} captions; skipped {missing_image_count} captions with missing/failed images."
    )

    if len(kept_caption_texts) == 0:
        logger.error("No captions remain after filtering missing images. Exiting.")
        raise RuntimeError("No captions remain after filtering")

    # 5) Encode captions (only the kept ones)
    caption_embeddings = encode_captions(
        text_model=text_model,
        captions=kept_caption_texts,
        device=device,
        batch_size=batch_size_text,
        logger=logger,
    )

    # 6) Prepare arrays and save .npz
    # Ensure dtypes: image indices int32
    caption_image_indices_arr = np.array(caption_image_indices, dtype=np.int32)
    images_names_arr = np.array(
        processed_filenames, dtype=object
    )  # Object dtype to allow variable-length strings

    # Metadata
    meta = {
        "metadata/num_captions": np.array(
            [caption_embeddings.shape[0]], dtype=np.int64
        ),
        "metadata/num_images": np.array([image_embeddings.shape[0]], dtype=np.int64),
        "metadata/embedding_dim_text": np.array(
            [caption_embeddings.shape[1]], dtype=np.int64
        ),
        "metadata/embedding_dim_image": np.array(
            [image_embeddings.shape[1]], dtype=np.int64
        ),
    }

    data = {
        **meta,
        "captions/embeddings": caption_embeddings,
        "captions/image_indices": caption_image_indices_arr,
        "images/names": images_names_arr,
        "images/embeddings": image_embeddings,
    }

    logger.info(f"Saving processed data to {output_file} (this may take some time)")
    np.savez_compressed(str(output_file), **data)
    logger.info("✓ Saved .npz file")
    return str(output_file)
