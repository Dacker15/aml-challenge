import sys
from pathlib import Path
import torch
import logging

from creation import create_npz_from_coco
from cleaning import filter_npz_file


def main():
    images_root = Path("data/coco2017/train2017")
    captions_json = Path("data/coco2017/annotations/captions_train2017.json")
    npz_file = Path("data/coco2017/coco_embeddings.npz")
    npz_clean_file = Path("data/coco2017/coco_embeddings_clean.npz")
    captions_file = Path("data/test/captions.txt")
    removed_images_file = Path("data/test/removed_images.txt")

    batch_size_image = 64  # If it goes Out Of Memory, lower to 32
    batch_size_text = 256

    # Device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("coco_to_npz")

    print("=" * 50)
    print("STARTING MS COCO PROCESSING")
    print(f"Images Folder: {images_root}")
    print(f"Captions File: {captions_json}")
    print(f"Output File:   {npz_file}")
    print(f"Device:        {device}")
    print(f"Image Batch Size:   {batch_size_image}")
    print(f"Text Batch Size:    {batch_size_text}")
    print("=" * 50)

    # Preliminary checks
    if not images_root.exists() or not images_root.is_dir():
        logger.error(f"ERROR: Image folder not found at: {images_root}")
        logger.error(
            "Make sure you have added the 'coco-2017-dataset' to your notebook."
        )
        sys.exit(2)
    if not captions_json.exists():
        logger.error(f"ERROR: Annotation file not found at: {captions_json}")
        sys.exit(2)

    # Pipeline execution
    try:
        create_npz_from_coco(
            images_root=images_root,
            captions_json=captions_json,
            output_file=npz_file,
            device=device,
            batch_size_image=batch_size_image,
            batch_size_text=batch_size_text,
            logger=logger,
        )
        print("\n" + "=" * 50)
        print(f"SUCCESS! File saved at: {npz_file.resolve()}")
        print("=" * 50)
        print("Starting dataset cleaning...")
        # Run filtering
        filter_npz_file(
            input_npz=npz_file,
            output_npz=npz_clean_file,
            captions_txt=captions_file,
            removed_images_txt=removed_images_file,
            logger=logger,
        )

    except Exception as e:
        logger.exception(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
