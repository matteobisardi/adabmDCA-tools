"""Flexible readers for Potts-model parameter files."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import torch

from adabmDCA.fasta import get_tokens, validate_alphabet


def load_params_flexible(
    fname: str | Path,
    tokens: str,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Load Potts-model parameters from plain-text or gzip-compressed files.

    This follows :func:`adabmDCA.load_params`, and also accepts gzip-compressed
    files and parameter files whose amino acids are stored as numeric state
    indices. Numeric state ``0`` is interpreted as the first supplied token
    (the gap state for the standard protein alphabet), ``1`` as the second,
    and so on.
    """
    path = Path(fname)
    with path.open("rb") as file:
        is_gzipped = file.read(2) == b"\x1f\x8b"

    opener = gzip.open if is_gzipped else open
    with opener(path, "rt") as file:
        lines = file.readlines()

    J_entries = []
    h_entries = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "J":
            J_entries.append((
                int(parts[1]), int(parts[2]), parts[3], parts[4], float(parts[5])
            ))
        elif parts[0] == "h":
            h_entries.append((int(parts[1]), parts[2], float(parts[3])))

    if not h_entries or not J_entries:
        raise ValueError("Parameter file must contain both h and J entries.")

    tokens = get_tokens(tokens)

    def state_index(state: str) -> int:
        try:
            index = int(state)
        except ValueError:
            if state not in tokens:
                validate_alphabet(np.array([state]), tokens=tokens)
            return tokens.index(state)
        if not 0 <= index < len(tokens):
            raise ValueError(
                f"Numeric state {index} is outside the supplied alphabet of "
                f"size {len(tokens)}."
            )
        return index

    h_idx0 = np.array([entry[0] for entry in h_entries], dtype=np.int64)
    h_idx1 = np.array([state_index(entry[1]) for entry in h_entries], dtype=np.int64)
    h_val = np.array([entry[2] for entry in h_entries], dtype=np.float64)
    J_idx0 = np.array([entry[0] for entry in J_entries], dtype=np.int64)
    J_idx1 = np.array([entry[1] for entry in J_entries], dtype=np.int64)
    J_idx2 = np.array([state_index(entry[2]) for entry in J_entries], dtype=np.int64)
    J_idx3 = np.array([state_index(entry[3]) for entry in J_entries], dtype=np.int64)
    J_val = np.array([entry[4] for entry in J_entries], dtype=np.float64)

    L = max(h_idx0.max(), J_idx0.max(), J_idx1.max()) + 1
    q = len(tokens)
    h = np.zeros((L, q), dtype=np.float64)
    h[h_idx0, h_idx1] = h_val
    J = np.zeros((L, L, q, q), dtype=np.float64)
    J[J_idx0, J_idx1, J_idx2, J_idx3] = J_val
    J = J + J.transpose(1, 0, 3, 2)
    J = J.transpose(0, 2, 1, 3)

    return {
        "bias": torch.tensor(h, dtype=dtype, device=device),
        "coupling_matrix": torch.tensor(J, dtype=dtype, device=device),
    }
