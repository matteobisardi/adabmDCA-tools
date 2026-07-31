"""Backward-compatible imports for the former combined functions module."""

from .fasta import import_from_fasta_keep_order, import_unaligned_fasta
from .metrics import (
    compute_gap_frequency,
    compute_pca,
    compute_ppv_contacts,
    compute_seqID,
    count_gaps,
    get_pairwise_seqid,
    inverse_one_hot,
)

__all__ = [
    "compute_gap_frequency",
    "compute_pca",
    "compute_ppv_contacts",
    "compute_seqID",
    "count_gaps",
    "get_pairwise_seqid",
    "import_from_fasta_keep_order",
    "import_unaligned_fasta",
    "inverse_one_hot",
]
