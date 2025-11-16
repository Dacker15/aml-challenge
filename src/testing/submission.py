import pandas as pd
import torch


def generate_submission(
    sample_ids, translated_embeddings, output_file="submission.csv"
):
    """
    Generate a submission.csv file from translated embeddings.
    """
    print("Generating submission file...")

    if isinstance(translated_embeddings, torch.Tensor):
        translated_embeddings = translated_embeddings.cpu().numpy()

    # Create a DataFrame with sample_id and embeddings

    df_submission = pd.DataFrame(
        {"id": sample_ids, "embedding": translated_embeddings.tolist()}
    )

    df_submission.to_csv(output_file, index=False, float_format="%.17g")

    return df_submission
