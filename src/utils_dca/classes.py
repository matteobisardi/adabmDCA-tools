"""Backward-compatible imports for the former combined classes module."""

from .dms import DeepMutationalScanning
from .metrics import compute_ppv_contacts
from .msa import MultipleSequenceAlignment
from .protein import ProteinSequence

__all__ = [
    "DeepMutationalScanning",
    "MultipleSequenceAlignment",
    "ProteinSequence",
    "compute_ppv_contacts",
]
