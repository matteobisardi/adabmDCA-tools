from __future__ import annotations

from collections.abc import MutableSequence
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from editdistance import eval as levenshtein_distance

from adabmDCA.fasta import encode_sequence
from adabmDCA.functional import one_hot

from .config import make_setup
from .fasta import import_unaligned_fasta


class ProteinSequence:
    def __init__(self, afa_sequence : str, setup=None, name : str = "GeneX" ):

        if setup is None:
            setup = make_setup()

        # Store setup before building the encoded representations.
        self._get_setup(setup)

        # Store sequences
        self.name = name
        self.afa = str(afa_sequence)
        self._dms = None
        self._refresh_from_afa()

        # Editable, 1-based views backed by this object rather than copies.
        self.aligned = _1ProteinSequence(self, "aligned")
        self.unaligned = _1ProteinSequence(self, "unaligned")

    @classmethod
    def from_path(cls, path, setup=None, name="GeneX"):
        """Create a protein sequence from a single-record aligned FASTA file.

        The FASTA sequence is passed to :class:`ProteinSequence` unchanged, so
        gaps and lowercase residues retain their alignment meaning. The FASTA
        header is used as the protein name unless ``name`` is provided.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to an aligned FASTA file containing exactly one sequence.
        setup : dict, optional
            Alphabet and compute setup passed to the constructor.
        name : str, optional
            Name for the protein. If omitted, use the FASTA header.

        Returns
        -------
        ProteinSequence
            The imported protein sequence.
        """
        headers, sequences = import_unaligned_fasta(
            path,
            tokens=None,
            filter_sequences=False,
            remove_duplicates=False,
        )
        if len(sequences) != 1:
            raise ValueError(
                "ProteinSequence.from_path() requires exactly one FASTA sequence; "
                f"found {len(sequences)}."
            )
        protein_name = str(headers[0]) if name is None else name
        return cls(sequences[0], setup=setup, name=protein_name)
        
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

    def _refresh_from_afa(self) -> None:
        """Rebuild every representation derived from ``afa``.

        Editing either public sequence view must keep the encoded tensor,
        position maps, and DMS helper in sync with the actual sequence.
        """
        self._aligned = self.remove_lowercase(self.afa)
        self._unaligned = self.remove_gaps(self.afa).upper()
        self._validate_sequence(self._aligned, allow_gaps=True)

        self.num = encode_sequence(self._aligned, self.tokens)
        self.enc = torch.tensor(self.num, device=self.device, dtype=torch.int32)
        self.onehot = one_hot(self.enc.view(1, -1), num_classes=self.q).to(self.dtype)

        self.L = len(self._aligned)
        self.L_unaligned = len(self._unaligned)
        self.a2u, self.u2a = self.map_positions(self.afa)

        # Existing DMS matrices refer to the previous sequence and are invalid.
        self._dms = None

    def _validate_sequence(self, sequence: str, *, allow_gaps: bool) -> None:
        allowed = set(self.tokens)
        if not allow_gaps:
            allowed.discard("-")
        invalid = set(sequence) - allowed
        if invalid:
            raise ValueError(
                f"Unknown token(s) {sorted(invalid)!r}; allowed tokens are {''.join(sorted(allowed))!r}."
            )

    def _replace_aligned(self, sequence: str) -> None:
        """Replace the aligned view and preserve deleted lowercase residues when possible."""
        sequence = str(sequence).upper()
        self._validate_sequence(sequence, allow_gaps=True)

        if len(sequence) == self.L:
            iterator = iter(sequence)
            self.afa = "".join(
                char if char.islower() else next(iterator) for char in self.afa
            )
        else:
            # A length-changing edit has no unambiguous placement among gaps
            # and deleted residues, so it becomes a new plain aligned sequence.
            self.afa = sequence
        self._refresh_from_afa()

    def _replace_unaligned(self, sequence: str) -> None:
        """Replace the unaligned view while preserving the alignment when possible."""
        sequence = str(sequence).upper()
        self._validate_sequence(sequence, allow_gaps=False)

        if len(sequence) == self.L_unaligned:
            iterator = iter(sequence)
            rebuilt = []
            for char in self.afa:
                if char == "-":
                    rebuilt.append(char)
                    continue
                replacement = next(iterator)
                rebuilt.append(replacement.lower() if char.islower() else replacement)
            self.afa = "".join(rebuilt)
        else:
            # Insertion/deletion in unaligned coordinates cannot be mapped to
            # existing alignment columns unambiguously.
            self.afa = sequence
        self._refresh_from_afa()
   
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
    """A 1-based mutable view backed by a :class:`ProteinSequence`."""
    def __init__(self, protein: ProteinSequence, kind: str):
        self._protein = protein
        self._kind = kind

    @property
    def _sequence(self) -> str:
        return self._protein._aligned if self._kind == "aligned" else self._protein._unaligned

    def _replace(self, sequence: str) -> None:
        if self._kind == "aligned":
            self._protein._replace_aligned(sequence)
        else:
            self._protein._replace_unaligned(sequence)

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
        return len(self._sequence)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._sequence[self._map_index(index)]
        elif isinstance(index, slice):
            return self._sequence[self._map_slice(index)]
        raise TypeError("Index must be int or slice")

    def __setitem__(self, index, value):
        seq = list(self._sequence)
        if isinstance(index, int):
            if not isinstance(value, str) or len(value) != 1:
                raise ValueError("Single-position assignment requires exactly one character.")
            seq[self._map_index(index)] = value
        elif isinstance(index, slice):
            if isinstance(value, str):
                replacement = value
            else:
                replacement = "".join(value)
            seq[self._map_slice(index)] = replacement
        else:
            raise TypeError("Index must be int or slice")
        self._replace("".join(seq))

    def __delitem__(self, index):
        seq = list(self._sequence)
        if isinstance(index, int):
            del seq[self._map_index(index)]
        elif isinstance(index, slice):
            del seq[self._map_slice(index)]
        else:
            raise TypeError("Index must be int or slice")
        self._replace("".join(seq))

    def insert(self, i, value):
        # insert BEFORE 1-based position i (like list.insert with 0-based)
        # i can be > len => behaves like list.insert
        if i is None:
            i = len(self) + 1
        if i == 0:
            raise IndexError("1-based indexing: position 0 is invalid")
        if not isinstance(value, str) or len(value) != 1:
            raise ValueError("Insertion requires exactly one character.")
        pos = self._map_index(i) if i > 0 else i
        seq = list(self._sequence)
        seq.insert(pos, value)
        self._replace("".join(seq))

    def get_string(self):
        return self._sequence


    # nice-to-haves
    def __iter__(self):
        return iter(self._sequence)

    def __repr__(self):
        return f'{self._sequence!r}'
