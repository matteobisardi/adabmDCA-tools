from __future__ import annotations

from collections.abc import MutableSequence
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from editdistance import eval as levenshtein_distance

from adabmDCA.fasta import encode_sequence
from adabmDCA.functional import one_hot

from .config import make_setup


class ProteinSequence:
    def __init__(self, afa_sequence : str, setup=None, name : str = "GeneX" ):

        if setup is None:
            setup = make_setup()

        # Store sequences
        self.name = name
        self.afa = afa_sequence
        self._aligned = self.remove_lowercase(self.afa) # aligned
        self._unaligned = self.remove_gaps(self.afa).upper() # unaligned

        # Store setup
        self._get_setup(setup)

        # Store encoding
        self.num = encode_sequence(self._aligned, self.tokens)
        self.enc = torch.tensor(self.num, device= self.device, dtype= torch.int32)
        self.onehot = one_hot(self.enc.view(1, -1), num_classes= self.q).to(self.dtype)


        # Attributes for user
        self.aligned = _1ProteinSequence(self._aligned)
        self.unaligned = _1ProteinSequence(self._unaligned)

        # Core sizes
        self.L: int = len(self._aligned)   # aligned length 
        self.L_unaligned: int = len(self._unaligned) # unaligned length 

        # Map positions
        self.a2u, self.u2a = self.map_positions(self.afa)

        # Initialize dms
        self._dms = None
        
    @property
    def dms(self):
        """Create the DMS helper only when it is first requested."""
        if self._dms is None:
            from .dms import DeepMutationalScanning
            self._dms = DeepMutationalScanning(self)
        return self._dms

    def __len__(self):
        return self.L

    # ---------- basic utilities ----------
    def _get_setup(self, setup):
        self.setup = setup
        self.device = self.setup["device"]
        self.dtype = self.setup["dtype"]
        self.tokens = self.setup["tokens"]
        self.q = self.setup["q"]
   
    @staticmethod
    def remove_gaps(seq: str) -> str:
        return "".join(ch for ch in seq if ch != "-")

    @staticmethod
    def remove_lowercase(seq: str) -> str:
        return "".join(ch for ch in seq if not ch.islower())

    def map_positions(self, afa_seq: str) -> Tuple[Dict[int, Optional[int]], Dict[int, Optional[int]]]:
        """
        Build 1-based maps between aligned and unaligned positions.

        a2u[aligned_col] -> unaligned_pos (or None for gaps/lowercase)
        u2a[unaligned_pos] -> aligned_col (or None if deleted).
        """

        Lu = self.L_unaligned
        a2u: Dict[int, Optional[int]] = {}
        u2a: Dict[int, Optional[int]] = {i + 1: None for i in range(Lu)}
        a_pos, u_pos = 0, 1
        for ch in afa_seq:
            if ch == "-":
                a_pos += 1
                a2u[a_pos] = None
            elif ch.islower():
                if u_pos <= Lu:
                    u2a[u_pos] = None
                    u_pos += 1
            else:  # UPPERCASE
                a_pos += 1
                if u_pos <= Lu:
                    a2u[a_pos] = u_pos
                    u2a[u_pos] = a_pos
                    u_pos += 1

        return a2u, u2a

    def hamming(self, other) -> int:
        """
        Hamming distance to another sequence (ProteinSequence, str, or 1D vector-like).
        """
        # Case 1: other is a ProteinSequence -> use aligned strings
        if isinstance(other, ProteinSequence):
            s1 = self._aligned
            s2 = other._aligned

        # Case 2: other is a string -> use aligned string vs provided string
        elif isinstance(other, str):
            s1 = self._aligned
            s2 = other

        # Case 3: other is a vector-like -> use encoded representation
        elif isinstance(other, (list, np.ndarray, torch.Tensor)):
            s1 = self.num
            if isinstance(other, torch.Tensor):
                s2 = other.detach().cpu().tolist()
            elif isinstance(other, np.ndarray):
                s2 = other.T
            else:  # list or other sequence
                s2 = list(other)

        else:
            raise TypeError(
                "other must be a ProteinSequence, a sequence string, "
                "or a 1D vector-like (list/np.ndarray/torch.Tensor)."
            )

        if len(s1) != len(s2):
            raise ValueError(
                f"Hamming distance requires equal-length inputs, "
                f"got {len(s1)} and {len(s2)}."
            )

        return sum(a != b for a, b in zip(s1, s2))

    def editd(self, other) -> int:
        """
        Levenshtein distance between unaligned sequences.

        other may be a ProteinSequence or a plain string.
        """
        # Case 1: other is a ProteinSequence -> use unaligned strings
        if isinstance(other, ProteinSequence):
            a = self._unaligned
            b = other._unaligned

        # Case 2: other is a string -> use unaligned string vs provided string
        elif isinstance(other, str):
            a = self._unaligned
            b = other

        else:
            raise TypeError(
                "other must be a ProteinSequence, a sequence string" )

        return levenshtein_distance(a, b)

class _1ProteinSequence(MutableSequence):
    """1-based view over a mutable sequence (e.g., list)."""
    def __init__(self, seq):
        self._seq = list(seq)

    # --- indexing helpers ---
    @staticmethod
    def _map_index(i: int) -> int:
        if i == 0:
            raise IndexError("1-based indexing: index 0 is invalid")
        return i-1 if i > 0 else i   # keep negative indices as-is

    @staticmethod
    def _map_slice(s: slice) -> slice:
        def adj(x):
            if x is None:
                return None
            if x == 0:
                raise IndexError("1-based indexing: slice bound 0 is invalid")
            return x-1 if x > 0 else x
        return slice(adj(s.start), s.stop, s.step)

    # --- core MutableSequence requirements ---
    def __len__(self):
        return len(self._seq)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._seq[self._map_index(index)]
        elif isinstance(index, slice):
            return "".join(self._seq[self._map_slice(index)])
        raise TypeError("Index must be int or slice")

    def __setitem__(self, index, value):
        if isinstance(index, int):
            self._seq[self._map_index(index)] = value
        elif isinstance(index, slice):
            self._seq[self._map_slice(index)] = value
        else:
            raise TypeError("Index must be int or slice")

    def __delitem__(self, index):
        if isinstance(index, int):
            del self._seq[self._map_index(index)]
        elif isinstance(index, slice):
            del self._seq[self._map_slice(index)]
        else:
            raise TypeError("Index must be int or slice")

    def insert(self, i, value):
        # insert BEFORE 1-based position i (like list.insert with 0-based)
        # i can be > len => behaves like list.insert
        if i is None:
            i = len(self) + 1
        if i == 0:
            raise IndexError("1-based indexing: position 0 is invalid")
        pos = self._map_index(i) if i > 0 else i
        self._seq.insert(pos, value)

    def get_string(self):
        return "".join(self._seq)


    # nice-to-haves
    def __iter__(self):
        return iter(self._seq)

    def __repr__(self):
        return f'{"".join(self._seq)!r}'

