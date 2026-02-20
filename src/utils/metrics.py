import torch
import torch.nn.functional as F


def embedding_variance(embeddings: torch.Tensor) -> float:
    """
    Compute the mean variance of embedding dimensions across the batch.

    This is the primary metric for detecting REPRESENTATION COLLAPSE — the
    biggest failure mode in SSL methods like JEPA/BYOL.

    What is representation collapse?
    ─────────────────────────────────
    If the student learns to map every input to the same (or very similar)
    vector, the MSE loss goes to zero trivially — but the model has learned
    nothing useful. This is called collapse. The EMA teacher and stop-gradient
    prevent it architecturally, but you should still monitor it.

    How to interpret the variance value:
        - Well-trained model:  variance > 0.01 (representations spread in space)
        - Early training:      variance ~0.001–0.01 (still learning to spread)
        - Collapse warning:    variance < 0.001 for many consecutive steps

    Args:
        embeddings : (N, D) tensor of embedding vectors

    Returns:
        float — mean per-dimension variance across the batch
    """
    if embeddings.shape[0] < 2:
        return 0.0

    # Variance per dimension (D,), then average across dimensions
    var = embeddings.var(dim=0).mean().item()
    return var


def cosine_similarity_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between two sets of vectors.

    Used in evaluation to check how semantically similar/dissimilar
    the model's representations are for different text inputs.

    Args:
        a : (N, D) — first set of embeddings
        b : (N, D) — second set of embeddings

    Returns:
        (N,) tensor of cosine similarities, one per pair
    """
    return F.cosine_similarity(a, b, dim=-1)


def mean_pairwise_cosine(embeddings: torch.Tensor) -> float:
    """
    Compute mean cosine similarity between ALL pairs in a batch.

    A well-trained encoder should produce diverse embeddings:
        - Semantically similar texts → high cosine sim (> 0.8)
        - Unrelated texts → low cosine sim (< 0.3)

    If mean pairwise cosine is very high (> 0.95) across ALL pairs
    (including unrelated texts), this indicates near-collapse.

    Args:
        embeddings : (N, D)

    Returns:
        float — mean cosine similarity across all pairs
    """
    # Normalize to unit sphere
    normed = F.normalize(embeddings, dim=-1)  # (N, D)

    # (N, N) similarity matrix
    sim_matrix = normed @ normed.T

    # Exclude diagonal (self-similarity = 1.0 always)
    N = sim_matrix.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool, device=embeddings.device)
    off_diag = sim_matrix[mask]

    return off_diag.mean().item()