"""Backward-compatible imports for the former combined classes module."""

from .dms import DeepMutationalScanning
from .metrics import compute_ppv_contacts
from .msa import MultipleSequenceAlignment
from .protein import ProteinSequence
from .sequence_path import SequencePath
from .sequence_path_fast import SequencePathFast

__all__ = [
    "DeepMutationalScanning",
    "MultipleSequenceAlignment",
    "ProteinSequence",
    "SequencePath",
    "SequencePathFast",
    "compute_ppv_contacts",
]
