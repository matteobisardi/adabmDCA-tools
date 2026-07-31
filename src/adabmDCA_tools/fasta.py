from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from adabmDCA.fasta import encode_sequence, get_tokens


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


