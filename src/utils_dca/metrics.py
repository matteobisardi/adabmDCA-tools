from typing import Dict, Optional

import numpy as np
import torch

from adabmDCA.utils import resample_sequences


def compute_gap_frequency(msa_oh: torch.Tensor):
    """Computes the frequency of gaps at each position in a multiple sequence alignment (MSA).

    Args:
        msa_oh (torch.Tensor): A one-hot encoded MSA tensor of shape (N, L, C), where:
            - N is the number of sequences.
            - L is the sequence length.
            - C is the number of possible characters, with index 0 representing gaps.

    Returns:
        torch.Tensor: A 1D tensor of shape (L,), representing the gap frequency at each position.
    """
    gap_frequencies = msa_oh[:, :, 0].mean(dim=0)
    return gap_frequencies

def compute_seqID(a1: torch.Tensor, single_seq: torch.Tensor):
    """
    Computes the Hamming distance 
    between a set of one-hot encoded sequences and a single one-hot encoded sequence.

    Args:
        a1 (torch.Tensor): Sequence dataset, shape (N, L, C), where N is the number of sequences,
                           L is the length, and C is the number of categories (one-hot size).
        single_seq (torch.Tensor): Single one-hot encoded sequence, shape (L, C).

    Returns:
        torch.Tensor: Hamming distances for each sequence in the dataset.
    """
    # print(a1.shape, single_seq.shape)
    a1 = a1.view(a1.shape[0], -1)
    single_seq = single_seq.view(1, -1)
    # print(a1.shape, single_seq.shape)
    seqID = (a1 * single_seq).sum(1) 

    return seqID

def get_pairwise_seqid(
    s1: torch.Tensor,
    s2: Optional[torch.Tensor] = None,
    unique: Optional[bool] = None,
    as_matrix: bool = False,
) -> np.ndarray:
    """
    Compute pairwise sequence identities from one-hot encoded sequences.

    The returned values are counts of matching alignment columns, not fractions.

    Args:
        s1: One-hot tensor with shape (N, L, q), or a single sequence (L, q).
        s2: One-hot tensor with shape (M, L, q), or a single sequence (L, q).
            If omitted, compares s1 with itself.
        unique: If True, return only the upper triangle excluding the diagonal.
            If None, this is enabled only when s2 is omitted or s1 and s2 are
            the same object.
        as_matrix: If True, return the full (N, M) matrix.

    Returns:
        A numpy array containing the pairwise sequence identities. For a
        singleton-vs-alignment comparison, the flattened result has length
        equal to the number of sequences in the other alignment.
    """
    same_input = s2 is None or s1 is s2
    if s2 is None:
        s2 = s1

    if s1.dim() == 2:
        s1 = s1.unsqueeze(0)
    elif s1.dim() != 3:
        raise ValueError("s1 must have shape (N, L, q) or (L, q).")

    if s2.dim() == 2:
        s2 = s2.unsqueeze(0)
    elif s2.dim() != 3:
        raise ValueError("s2 must have shape (M, L, q) or (L, q).")

    if s1.shape[1:] != s2.shape[1:]:
        raise ValueError(
            f"Shape mismatch: s1 has shape {tuple(s1.shape)}, "
            f"s2 has shape {tuple(s2.shape)}."
        )

    s2 = s2.to(device=s1.device, dtype=s1.dtype)
    s1_flat = s1.reshape(s1.shape[0], -1)
    s2_flat = s2.reshape(s2.shape[0], -1)
    pairwise = s1_flat @ s2_flat.T

    if as_matrix:
        return pairwise.cpu().numpy()

    pairwise_cpu = pairwise.cpu()

    if unique is None:
        unique = same_input

    if unique:
        if pairwise_cpu.shape[0] != pairwise_cpu.shape[1]:
            raise ValueError("unique=True requires a square self-comparison matrix.")
        i, j = torch.triu_indices(
            pairwise_cpu.shape[0],
            pairwise_cpu.shape[1],
            offset=1,
        )
        return pairwise_cpu[i, j].numpy()

    return pairwise_cpu.flatten().numpy()

def inverse_one_hot(one_hot_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a one-hot encoded tensor back to its original class indices.
    
    Args:
        one_hot_tensor (torch.Tensor): One-hot encoded tensor of shape (N, C) where
                                       N is the number of samples and C is the number of classes.
                                       
    Returns:
        torch.Tensor: Tensor of shape (N,) containing the original class indices.
    """
    return torch.argmax(one_hot_tensor, dim=-1)

# Define a function that counts the number of gaps in an MSA
def count_gaps(msa: torch.Tensor):
    """
    Counts the number of gaps in each sequence of a multiple sequence alignment (MSA) or a single sequence.

    Args:
        msa (torch.Tensor): A one-hot encoded MSA tensor of shape (N, L, C) or a single sequence tensor of shape (L, C), where:
            - N is the number of sequences.
            - L is the sequence length.
            - C is the number of possible characters, with index 0 representing gaps.

    Returns:
        torch.Tensor: A 1D tensor representing the number of gaps in each sequence if input is MSA, or a scalar tensor if input is a single sequence.
    """
    if msa.dim() == 2:
        # Single sequence case
        return torch.sum(msa[:, 0])
    elif msa.dim() == 3:
        # MSA case
        return torch.sum(msa[:, :, 0], dim=1)
    else:
        raise ValueError("Input tensor must be of shape (N, L, C) or (L, C)")

def compute_pca(
    msa_oh: torch.Tensor,
    weights: torch.Tensor | None = None,
    n_components: int = 4,
) -> torch.Tensor:
    """
    Compute PCA on an MSA one-hot tensor.

    Parameters
    ----------
    msa_oh : torch.Tensor
        Tensor of shape (M, L, q), one-hot encoding of the alignment.
    weights : torch.Tensor, optional
        Sequence weights, shape (M,).

    Returns
    -------
    torch.Tensor
        Projections on the requested PCs, shape (M, n_components).
    """
    print(
        "Do you need the standalone compute_pca()? "
        "MultipleSequenceAlignment.compute_pca() provides the same operation."
    )

    if n_components < 1:
        raise ValueError("n_components must be at least 1.")

    M, L, q = msa_oh.shape
    device = msa_oh.device
    dtype = msa_oh.dtype

    if weights is None:
        weights = torch.ones(M, device=device, dtype=msa_oh.dtype)

    msa_mix = resample_sequences(msa_oh, weights=weights, nextract=M)
    msa_flat = msa_mix.view(M, -1)

    # Center for SVD
    centered = msa_flat - msa_flat.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)

    msa_proj = (centered @ Vh.T)[:, :n_components] / (L ** 0.5)

    return msa_proj


def compute_ppv_contacts(
    dca_scores: np.ndarray,
    ca_distances: np.ndarray,
    *,
    dist_cutoff: float = 8.0,
    min_seq_sep: int = 4,
    top_k: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute PPV (precision) for DCA contact prediction against real Ca-Ca distances.

    Only the upper triangle is used (i < j) and positions with |i-j| > min_seq_sep.

    Parameters
    ----------
    dca_scores : (L, L) array
        DCA score matrix (higher means more likely contact).
    ca_distances : (L, L) array
        Ca-Ca distance matrix in Angstrom.
    dist_cutoff : float
        Distance threshold to define a true contact.
    min_seq_sep : int
        Minimum sequence separation: only pairs with |i-j| > min_seq_sep are considered.
    top_k : int or None
        Number of top-scoring pairs to predict. If None, uses the number of true
        contacts among the considered pairs.

    Returns
    -------
    dict
        {"ppv": ..., "tp": ..., "fp": ..., "n_pred": ..., "n_true": ...}
    """
    scores = np.asarray(dca_scores, dtype=float)
    distances = np.asarray(ca_distances, dtype=float)

    if scores.shape != distances.shape or scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("dca_scores and ca_distances must be square matrices with the same shape.")

    L = scores.shape[0]
    # Upper triangle with sequence separation constraint (j - i > min_seq_sep)
    mask = np.triu(np.ones((L, L), dtype=bool), k=min_seq_sep + 1)

    scores_vec = scores[mask]
    true_vec = distances[mask] <= dist_cutoff

    n_true = int(true_vec.sum())
    if top_k is None:
        top_k = n_true

    top_k = int(top_k)
    if top_k < 0 or top_k > scores_vec.size:
        raise ValueError("top_k must be between 0 and the number of considered pairs.")

    if top_k == 0:
        return {"ppv": float("nan"), "tp": 0.0, "fp": 0.0, "n_pred": 0.0, "n_true": float(n_true)}

    top_idx = np.argsort(scores_vec)[::-1][:top_k]
    tp = int(true_vec[top_idx].sum())
    fp = int(top_k - tp)
    ppv = tp / top_k

    return {"ppv": float(ppv), "tp": float(tp), "fp": float(fp), "n_pred": float(top_k), "n_true": float(n_true)}


