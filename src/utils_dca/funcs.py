import torch
import pandas
from pathlib import Path
from typing import Tuple, Literal, List, Optional
import numpy as np
import adabmDCA
from adabmDCA.fasta import *
from adabmDCA.utils import *

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
 

def import_from_fasta_keep_order(
    fasta_name: str | Path,
    tokens: str | None = None,
    filter_sequences: bool = False,
    remove_duplicates: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Import sequences from a fasta file. The following operations are performed:
    - If 'tokens' is provided, encodes the sequences in numeric format.
    - If 'filter_sequences' is True, removes the sequences whose tokens are not present in the alphabet.
    - If 'remove_duplicates' is True, removes duplicated sequences while keeping the first occurrence order.
    """
    # Import headers and sequences
    sequences = []
    names = []
    seq = ''
    with open(fasta_name, 'r') as f:
        first_line = f.readline()
        if not first_line.startswith('>'):
            raise RuntimeError(f"The file {fasta_name} is not in a fasta format.")
        f.seek(0)
        for line in f:
            if not line.strip():
                continue
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                header = line[1:].strip()
                names.append(header)
                seq = ''
            else:
                seq += line.strip()
    if seq:
        sequences.append(seq)
    
    # Filter sequences
    if filter_sequences:
        if tokens is None:
            raise ValueError("Argument 'tokens' must be provided if 'filter_sequences' is True.")
        tokens = get_tokens(tokens)
        tokens_list = [a for a in tokens]
        clean_names = []
        clean_sequences = []
        for n, s in zip(names, sequences):
            good_sequence = np.full(shape=(len(s),), fill_value=False)
            splitline = np.array([a for a in s])
            for token in tokens_list:
                good_sequence += (token == splitline)
            if np.all(good_sequence):
                if n == "":
                    n = "unknown_sequence"
                clean_names.append(n)
                clean_sequences.append(s)
            else:
                print(f"Unknown token found: removing sequence {n}")
        names = np.array(clean_names)
        sequences = np.array(clean_sequences)
        
    else:
        names = np.array(names)
        sequences = np.array(sequences)
    
    # Remove duplicates while preserving the original sequence order.
    if remove_duplicates:
        _, unique_ids = np.unique(sequences, return_index=True)
        unique_ids = np.sort(unique_ids)
        sequences = sequences[unique_ids]
        names = names[unique_ids]
    
    if (tokens is not None) and (len(sequences) > 0):
        sequences = encode_sequence(sequences, tokens)
    
    return names, sequences



def import_unaligned_fasta(
    fasta_name: str | Path,
    tokens: Optional[str] = None,
    filter_sequences: bool = False,
    remove_duplicates: bool = True,
    ) -> Tuple[List[str], List[str]]:
    """Import unaligned sequences from a FASTA file.

    Args:
        fasta_name: Path to the FASTA file.
        tokens: Optional string of allowed characters (e.g., 'ACDEFGHIKLMNPQRSTVWY').
                Used only if filter_sequences=True.
        filter_sequences: If True, drop sequences containing characters not in `tokens`.
        remove_duplicates: If True, drop exact duplicate sequence strings (keep first).

    Returns:
        (headers, sequences): two lists of equal length.

    Raises:
        RuntimeError: If the file doesn't look like FASTA (first non-empty line not starting with '>').
        ValueError: If filter_sequences=True but tokens is None.
    """
    fasta_path = Path(fasta_name)
    if not fasta_path.exists():
        raise FileNotFoundError(f"No such file: {fasta_path}")

    headers: List[str] = []
    sequences: List[str] = []

    # Parse FASTA
    seq_chunks: List[str] = []
    current_header: Optional[str] = None

    with open(fasta_path, "r") as f:
        # Validate FASTA by checking first non-empty line
        for first_line in f:
            if first_line.strip():
                if not first_line.startswith(">"):
                    raise RuntimeError(f"The file {fasta_path} is not in FASTA format.")
                # Rewind to start for a clean parse
                f.seek(0)
                break

        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # Flush previous record
                if current_header is not None:
                    headers.append(current_header)
                    sequences.append("".join(seq_chunks))
                    seq_chunks.clear()
                current_header = line[1:].strip() or "unknown_sequence"
            else:
                seq_chunks.append(line)

        # Flush last record
        if current_header is not None:
            headers.append(current_header)
            sequences.append("".join(seq_chunks))

    # Optional filtering by allowed tokens
    if filter_sequences:
        if tokens is None:
            raise ValueError("Argument 'tokens' must be provided if 'filter_sequences' is True.")
        allowed = set(tokens)
        kept_headers: List[str] = []
        kept_sequences: List[str] = []
        for h, s in zip(headers, sequences):
            if set(s).issubset(allowed):
                kept_headers.append(h)
                kept_sequences.append(s)
            # else: silently drop; print or log if desired
        headers, sequences = kept_headers, kept_sequences

    # Optional duplicate removal (preserve first occurrence)
    if remove_duplicates:
        seen = set()
        dedup_headers: List[str] = []
        dedup_sequences: List[str] = []
        for h, s in zip(headers, sequences):
            if s not in seen:
                seen.add(s)
                dedup_headers.append(h)
                dedup_sequences.append(s)
        headers, sequences = dedup_headers, dedup_sequences

    return headers, sequences

def compute_pca(msa_oh: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute PCA on an MSA one-hot tensor and return the first 2 components.

    Parameters
    ----------
    msa_oh : torch.Tensor
        Tensor of shape (M, L, q), one-hot encoding of the alignment.
    weights : torch.Tensor, optional
        Sequence weights, shape (M,).

    Returns
    -------
    torch.Tensor
        Projections on the first 2 PCs, shape (M, 2).
    """
    M, L, q = msa_oh.shape
    device = msa_oh.device
    dtype = msa_oh.dtype

    if weights is None:
        weights = torch.ones(M, device=device, dtype=msa_oh.dtype)

    msa_mix = resample_sequences(msa_oh, weights=weights, nextract=M)
    msa_flat = msa_mix.view(M, -1)

    # Center for SVD
    _, _, Vt = torch.svd(msa_flat - msa_flat.mean(0, keepdim=True))

    # Project without centering, as in your snippet
    msa_proj = msa_flat @ Vt / (L ** 0.5)

    return msa_proj
