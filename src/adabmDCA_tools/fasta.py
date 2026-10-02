from pathlib import Path
from typing import List, Optional, Tuple
import gzip
from collections import Counter

import numpy as np
from Bio import SeqIO

from adabmDCA.fasta import encode_sequence, get_tokens


def import_from_fasta_keep_order(
    fasta_name: str | Path,
    tokens: str | None = None,
    filter_sequences: bool = False,
    remove_duplicates: bool = True,
    unknown_token_policy: str = "remove",
) -> Tuple[np.ndarray, np.ndarray]:
    """Import sequences from a fasta file. The following operations are performed:
    - If 'tokens' is provided, encodes the sequences in numeric format.
    - If 'filter_sequences' is True, removes sequences with unknown tokens by default.
      With 'replace_with_gap', replaces each unknown token with '-' instead.
    - If 'remove_duplicates' is True, removes duplicated sequences while keeping the first occurrence order.
    """
    if unknown_token_policy not in {"remove", "replace_with_gap"}:
        raise ValueError("unknown_token_policy must be 'remove' or 'replace_with_gap'.")

    # Follow adabmDCA's SeqIO reader, including gzip-compressed FASTA files.
    opener = gzip.open if str(fasta_name).endswith('.gz') else open
    with opener(fasta_name, 'rt') as fasta_file:
        records = list(SeqIO.parse(fasta_file, 'fasta'))
    names = np.array([str(record.description) for record in records])
    sequences = np.array([str(record.seq) for record in records])

    # Some aligned protein FASTAs append a translation stop marker after the
    # alignment. Remove it only when that record is exactly one column longer
    # than the most common alignment length. An internal '*' remains a gap.
    if filter_sequences and unknown_token_policy == "replace_with_gap" and len(sequences):
        common_length = Counter(map(len, sequences)).most_common(1)[0][0]
        sequences = np.array([
            s[:-1] if len(s) == common_length + 1 and s.endswith('*') else s
            for s in sequences
        ])

    # Filter or repair unknown tokens before deduplication and encoding.
    if filter_sequences:
        if tokens is None:
            raise ValueError("Argument 'tokens' must be provided if 'filter_sequences' is True.")
        tokens = get_tokens(tokens)
        if unknown_token_policy == "replace_with_gap" and '-' not in tokens:
            raise ValueError("'replace_with_gap' requires '-' in the token alphabet.")
        allowed = set(tokens)
        clean_names = []
        clean_sequences = []
        for n, s in zip(names, sequences):
            if set(s) <= allowed:
                clean_names.append(n)
                clean_sequences.append(s)
            elif unknown_token_policy == "replace_with_gap":
                clean_names.append(n)
                clean_sequences.append(''.join(a if a in allowed else '-' for a in s))
            else:
                print(f"Unknown token found: removing sequence {n}")
        names = np.array(clean_names)
        sequences = np.array(clean_sequences)

    # Remove duplicates while preserving the original sequence order.
    if remove_duplicates:
        _, unique_ids = np.unique(sequences, return_index=True)
        unique_ids = np.sort(unique_ids)
        sequences = sequences[unique_ids]
        names = names[unique_ids]

    if len(sequences) and len({len(s) for s in sequences}) != 1:
        lengths = Counter(map(len, sequences))
        examples = [(str(n), len(s)) for n, s in zip(names, sequences)
                    if len(s) != lengths.most_common(1)[0][0]][:5]
        raise ValueError(
            f"FASTA sequences have unequal aligned lengths: {dict(lengths)}. "
            f"Examples: {examples}. Unknown-token replacement does not pad sequences."
        )

    if (tokens is not None) and (len(sequences) > 0):
        sequences = encode_sequence(sequences, tokens)
    
    return names, sequences



def import_unaligned_fasta(
    fasta_name: str | Path,
    tokens: Optional[str] = None,
    filter_sequences: bool = False,
    remove_duplicates: bool = False,
    ) -> Tuple[List[str], List[str]]:
    """Import unaligned sequences from a FASTA file.

    Args:
        fasta_name: Path to the FASTA file.
        tokens: Optional string of allowed characters (e.g., 'ACDEFGHIKLMNPQRSTVWY').
                Used only if filter_sequences=True.
        filter_sequences: If True, drop sequences containing characters not in `tokens`.
        remove_duplicates: If True, drop exact duplicate sequence strings (keep first).
            The default False preserves every record from the input FASTA.

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
