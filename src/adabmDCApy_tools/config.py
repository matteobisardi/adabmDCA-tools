"""Configuration helpers for :mod:`adabmDCApy_tools`."""

import torch

from adabmDCA.fasta import get_tokens
from adabmDCA.utils import get_device, get_dtype


def make_setup(
    alphabet: str = "protein",
    device: str = "auto",
    dtype: str = "float32",
) -> dict:
    """Create the configuration expected by ``adabmDCApy_tools`` classes.

    Parameters
    ----------
    alphabet:
        ``"protein"``, ``"rna"``, ``"dna"``, or a custom token string.
    device:
        ``"auto"`` selects CUDA, then MPS, then CPU. Explicit values
        ``"cpu"``, ``"cuda"``, and ``"mps"`` are also accepted.
    dtype:
        Tensor precision understood by ``adabmDCA.utils.get_dtype``.
    """
    tokens = get_tokens(alphabet)

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    return {
        "device": get_device(device, message=False),
        "dtype": get_dtype(dtype),
        "tokens": tokens,
        "q": len(tokens),
    }
