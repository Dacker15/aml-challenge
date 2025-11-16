import numpy as np
import torch


def mrr(
    translated_embd: torch.Tensor | np.ndarray,
    image_embd: torch.Tensor | np.ndarray,
    gt_indices: torch.Tensor | np.ndarray,
    max_indices=99,
    batch_size=100,
):
    """
    Compute Mean Reciprocal Rank (MRR) in batches to save memory
    Args:
        translated_embd: (N, D) tensor/array of translated embeddings
        image_embd: (M, D) tensor/array of image embeddings
        gt_indices: (N,) tensor/array of ground truth indices
        max_indices: maximum number of top indices to consider
        batch_size: batch size for processing
    Returns:
        mrr: Mean Reciprocal Rank
    """
    if isinstance(translated_embd, np.ndarray):
        translated_embd = torch.from_numpy(translated_embd).float()
    if isinstance(image_embd, np.ndarray):
        image_embd = torch.from_numpy(image_embd).float()
    if isinstance(gt_indices, np.ndarray):
        gt_indices = torch.from_numpy(gt_indices)

    n_queries = translated_embd.shape[0]
    all_sorted_indices = []
    gt_indices_cpu = gt_indices.cpu().numpy()

    for start_idx in range(0, n_queries, batch_size):
        batch_slice = slice(start_idx, min(start_idx + batch_size, n_queries))
        batch_translated = translated_embd[batch_slice]
        batch_similarity = batch_translated @ image_embd.T

        current_batch_k = batch_similarity.shape[1]
        k_to_use = min(max_indices, current_batch_k)
        batch_indices = (
            batch_similarity.topk(k=k_to_use, dim=1, sorted=True).indices.cpu().numpy()
        )

        ranks = []
        for i in range(len(batch_indices)):
            target = gt_indices_cpu[batch_slice][i]
            preds = batch_indices[i]
            try:
                rank = np.where(preds == target)[0][0] + 1
                ranks.append(1.0 / rank)
            except IndexError:
                ranks.append(0.0)
        all_sorted_indices.extend(ranks)

    return np.mean(all_sorted_indices)
