from __future__ import annotations

import gc

import numpy as np
import torch

from adabmDCA.fasta import compute_weights, encode_sequence, write_fasta
from adabmDCA.functional import one_hot
from adabmDCA.statmech import compute_energy
from adabmDCA.utils import resample_sequences

from .config import make_setup
from .fasta import import_from_fasta_keep_order


class MultipleSequenceAlignment:
    """A multiple-sequence alignment with memory-conscious tensor storage.

    ``enc`` is the single persistent numerical representation. It always stays
    on CPU, where it is compact (one integer per alignment position). One-hot
    tensors are created only by :meth:`get_onehot` when a calculation needs
    them, because a protein one-hot representation is about 21 times larger.
    """

    def __init__(self, headers, seqs, setup=None):
        if setup is None:
            setup = make_setup()
        self._get_setup(setup)

        self.headers = np.asarray(headers)

        # CPU encoded data are the source of truth. Keeping this on CPU avoids
        # permanently occupying MPS memory merely because an MSA was imported.
        self.enc = torch.tensor(seqs, device="cpu", dtype=torch.int32).contiguous()
        if self.enc.dim() != 2:
            raise ValueError("seqs must be an encoded array with shape (M, L).")

        # The one-hot representation is an optional explicit cache, never
        # created on import.  Keep it private so ``onehot()`` can be the
        # public method name.
        self._onehot_cache = None
        self._onehot_device = None
        self._V = None
        self._pca_mean = None

        self.M, self.L = self.enc.shape
        self.Meff = None
        self.gap_freq = None
        self.weights = None  # Small (M,) CPU tensor when computed.

        self.family = ""
        self.author = ""
        self.name = ""

    @property
    def seqs(self) -> np.ndarray:
        """CPU NumPy view of the encoded alignment.

        This is a view of ``enc`` rather than a second persistent copy. Do not
        modify it in place; use :meth:`remove_items` so derived statistics and
        caches can be invalidated correctly.
        """
        return self.enc.numpy()

    def __len__(self):
        return self.M

    def __getitem__(self, idx: int):
        return self.headers[idx], self.seqs[idx]

    # ---------------- #
    # -- Utilities -- #
    def _get_setup(self, setup):
        # Copy the dictionary: changing this MSA's compute device must not
        # silently change every other object that received the same setup.
        self.setup = dict(setup)
        self.device = torch.device(self.setup["device"])
        self.setup["device"] = self.device
        self.dtype = self.setup["dtype"]
        self.tokens = self.setup["tokens"]
        self.q = self.setup["q"]

    def _resolve_device(self, device=None) -> torch.device:
        return self.device if device is None else torch.device(device)

    def to(self, device):
        """Set the default *temporary computation* device and return ``self``.

        The stored alignment remains on CPU. Existing PCA and one-hot caches
        are dropped because they belong to the previous computation device.
        """
        self.device = torch.device(device)
        self.setup["device"] = self.device
        self.clear_cache()
        return self

    def onehot(self, device=None, *, cache=False) -> torch.Tensor:
        """Create a one-hot alignment on ``device``.

        Parameters
        ----------
        device:
            Target device. If omitted, use the device selected by
            :func:`make_setup` (available as ``self.device``).
        cache:
            If True, keep this tensor internally for reuse. The default False
            returns a temporary tensor and leaves no package-held cache.

        Notes
        -----
        A caller retaining the returned tensor also retains its memory. For a
        large temporary MPS tensor, use ``del tensor`` after use and then call
        ``msa.clear_cache()`` if needed.
        """
        target = self._resolve_device(device)
        if cache and self._onehot_cache is not None and self._onehot_device == target:
            return self._onehot_cache
        if cache and self._onehot_cache is not None:
            # Do not keep two potentially huge cached representations while
            # changing devices.
            self._onehot_cache = None
            self._onehot_device = None

        encoded = self.enc if target.type == "cpu" else self.enc.to(target)
        result = one_hot(encoded, num_classes=self.q).to(dtype=self.dtype)

        if cache:
            self._onehot_cache = result
            self._onehot_device = target
        return result

    def get_onehot(self, device=None, *, cache=False) -> torch.Tensor:
        """Compatibility alias for :meth:`onehot`."""
        return self.onehot(device=device, cache=cache)

    def clear_cache(self, *, empty_mps_cache=True):
        """Release package-held one-hot and PCA tensors.

        This does not delete tensors held by notebook variables outside this
        object. It only releases caches owned by the MSA itself.
        """
        self._onehot_cache = None
        self._onehot_device = None
        self._V = None
        self._pca_mean = None
        gc.collect()
        if (
            empty_mps_cache
            and hasattr(torch, "mps")
            and torch.backends.mps.is_available()
        ):
            torch.mps.empty_cache()

    # ---------------- #
    # -- Import MSA -- #
    @classmethod
    def from_path(cls, path, setup=None, remove_duplicates=False):
        if setup is None:
            setup = make_setup()
        headers, seqs = import_from_fasta_keep_order(
            path,
            setup["tokens"],
            filter_sequences=True,
            remove_duplicates=remove_duplicates,
        )
        return cls(headers, seqs, setup)

    @classmethod
    def from_onehot(cls, msa_oh, setup=None):
        if not isinstance(msa_oh, torch.Tensor) or msa_oh.dim() != 3:
            raise ValueError("msa_oh must be a tensor with shape (M, L, q).")
        if setup is None:
            setup = make_setup()
        if msa_oh.shape[-1] != setup["q"]:
            raise ValueError(
                f"One-hot alphabet size q={msa_oh.shape[-1]} does not match "
                f"setup q={setup['q']}. Pass an explicit setup for RNA, DNA, "
                "or custom alphabets."
            )
        seqs = torch.argmax(msa_oh, dim=-1).cpu().numpy()
        headers = [f"seq{i}" for i in range(1, len(seqs) + 1)]
        return cls(headers, seqs, setup)

    # ---------------- #
    # -- Statistics -- #
    def compute_weights_cls(self, th=0.8, device=None):
        """Compute weights from compact encodings and keep only the CPU result."""
        target = self._resolve_device(device)
        encoded = self.enc if target.type == "cpu" else self.enc.to(target)
        self.weights = compute_weights(
            encoded,
            th=th,
            device=target,
            dtype=self.dtype,
        ).cpu()
        self.Meff = float(self.weights.sum())
        return self.weights

    def compute_gap_frequency(self, th=0.8):
        """Compute sequence-weighted gap frequencies for each alignment column.

        Parameters
        ----------
        th : float, default=0.8
            Sequence-identity threshold used to compute the sequence weights.

        Returns
        -------
        torch.Tensor
            A CPU tensor of shape ``(L,)``. Each column is the weighted sum of
            its gap indicators divided by the total sequence weight.
        """
        # The gap statistic and sequence weights must be computed from the same
        # identity threshold.  ``compute_weights_cls`` keeps only the compact
        # CPU weight vector after doing any requested accelerator work.
        weights = self.compute_weights_cls(th=th)
        total_weight = weights.sum()
        if total_weight <= 0:
            raise ValueError("The total sequence weight must be positive.")

        # PCA statistics are computed from the sequence weights, so changing
        # the identity threshold invalidates any cached weighted PCA basis.
        self._V = None
        self._pca_mean = None
        gap_indicator = (self.enc == 0).to(dtype=weights.dtype)
        self.gap_freq = (weights[:, None] * gap_indicator).sum(dim=0) / total_weight
        return self.gap_freq

    def estimate_potts_snr(self, model, th=0.8, device=None):
        """Estimate the signal-to-noise ratio of an inferred Potts model.

        This uses the Haldane--Levy approximation

        ``SNR = Neff * Var[E(S)] / theta_Potts``,

        where ``theta_Potts = L(q - 1) + L(L - 1)(q - 1)^2 / 2``. The energy
        variance is the sequence-weighted population variance over this MSA,
        consistent with the effective sequence count used in the numerator.

        Parameters
        ----------
        model : dict
            adabmDCA Potts parameters containing ``bias`` with shape ``(L, q)``
            and ``coupling_matrix`` with shape ``(L, q, L, q)``.
        th : float, default=0.8
            Sequence-identity threshold used only when weights have not already
            been computed.
        device : str or torch.device, optional
            Device used for the temporary one-hot alignment and energy
            calculation. By default, use the model's device.

        Returns
        -------
        float
            Estimated Potts-model signal-to-noise ratio.
        """
        if not isinstance(model, dict):
            raise TypeError("model must be an adabmDCA parameter dictionary.")
        if "bias" not in model or "coupling_matrix" not in model:
            raise ValueError("model must contain 'bias' and 'coupling_matrix'.")

        bias = model["bias"]
        couplings = model["coupling_matrix"]
        if not isinstance(bias, torch.Tensor) or not isinstance(couplings, torch.Tensor):
            raise TypeError("model parameters must be torch tensors.")
        if bias.dim() != 2:
            raise ValueError("model['bias'] must have shape (L, q).")

        model_L, model_q = bias.shape
        if model_L != self.L or model_q != self.q:
            raise ValueError(
                "Model dimensions do not match the MSA: "
                f"got (L={model_L}, q={model_q}), "
                f"expected (L={self.L}, q={self.q})."
            )
        expected_coupling_shape = (self.L, self.q, self.L, self.q)
        if tuple(couplings.shape) != expected_coupling_shape:
            raise ValueError(
                "model['coupling_matrix'] must have shape "
                f"{expected_coupling_shape}."
            )

        target = bias.device if device is None else torch.device(device)
        params = {
            "bias": bias.detach().to(target),
            "coupling_matrix": couplings.detach().to(target, dtype=bias.dtype),
        }
        if self.weights is None or self.Meff is None:
            self.compute_weights_cls(th=th, device=target)
        if self.Meff <= 0:
            raise ValueError("Meff must be positive to estimate the SNR.")

        with torch.no_grad():
            msa_oh = self.onehot(device=target).to(dtype=params["bias"].dtype)
            energies = compute_energy(msa_oh, params).reshape(-1)
            weights = self.weights.to(target, dtype=energies.dtype)
            normalized_weights = weights / weights.sum()
            mean_energy = torch.sum(normalized_weights * energies)
            chi_squared = torch.sum(
                normalized_weights * torch.square(energies - mean_energy)
            )

        q_minus_one = self.q - 1
        theta_potts = (
            self.L * q_minus_one
            + self.L * (self.L - 1) * q_minus_one**2 / 2
        )
        return float(self.Meff * chi_squared.item() / theta_potts)

    def recompute_statistics(self, fast=False, th=0.8):
        """Refresh derived MSA statistics after changing its sequences.

        By default, recompute sequence weights, ``Meff``, and weighted gap
        frequencies. Set ``fast=True`` only when batching several edits and
        planning to call this method again with ``fast=False`` afterwards.
        """
        self.weights = None
        self.Meff = None
        self.gap_freq = None
        self.clear_cache()
        self.M, self.L = self.enc.shape
        if not fast:
            # Gap frequencies are sequence-weighted, so recomputing them also
            # recomputes the weights using the supplied identity threshold.
            self.compute_gap_frequency(th=th)

    def concatenate(self, other, *, fast=False, th=0.8):
        """Append the sequences and headers of another MSA in place.

        Both alignments must have the same length and alphabet. The encoded
        alignment and headers are always joined. Sequence weights, weighted gap
        frequencies, ``Meff``, and PCA state depend on the complete population
        and are invalidated.

        Parameters
        ----------
        other : MultipleSequenceAlignment
            Alignment whose rows are appended to this alignment.
        fast : bool, default=False
            If False, recompute weighted gap frequencies and sequence weights
            for the concatenated alignment. Set True to defer this work.
        th : float, default=0.8
            Sequence-identity threshold used for weighted gap frequencies when
            ``fast=False``.

        Returns
        -------
        MultipleSequenceAlignment
            ``self``, after concatenation.
        """
        if not isinstance(other, MultipleSequenceAlignment):
            raise TypeError("other must be a MultipleSequenceAlignment.")
        if other is self:
            raise ValueError("An MSA cannot be concatenated with itself in place.")
        if other.L != self.L:
            raise ValueError(
                f"Alignment length mismatch: got L={other.L}, expected L={self.L}."
            )
        if other.q != self.q or other.tokens != self.tokens:
            raise ValueError("Both MSAs must use the same alphabet.")

        combined_onehot = None
        combined_onehot_device = None
        if (
            self._onehot_cache is not None
            and other._onehot_cache is not None
            and other._onehot_device == self._onehot_device
            and other._onehot_cache.dtype == self._onehot_cache.dtype
        ):
            combined_onehot = torch.cat(
                (self._onehot_cache, other._onehot_cache), dim=0
            ).contiguous()
            combined_onehot_device = self._onehot_device

        self.enc = torch.cat((self.enc, other.enc), dim=0).contiguous()
        self.headers = np.concatenate((self.headers, other.headers))
        self.M, self.L = self.enc.shape

        # Sequence weights depend on the complete concatenated population;
        # neither operand's weighted gap frequencies remain valid.
        self.gap_freq = None
        self.weights = None
        self.Meff = None
        self._V = None
        self._pca_mean = None
        self._onehot_cache = combined_onehot
        self._onehot_device = combined_onehot_device

        if not fast:
            self.compute_gap_frequency(th=th)
        return self

    @classmethod
    def concatenate_many(cls, *alignments, fast=False, th=0.8):
        """Create a new MSA by concatenating one or more alignments.

        The input alignments are not modified. See :meth:`concatenate` for the
        cache-preservation rules and the meaning of ``fast`` and ``th``.

        Parameters
        ----------
        *alignments : MultipleSequenceAlignment
            Alignments to concatenate, in order.
        fast : bool, default=False
            If False, compute statistics for the concatenated alignment. Set
            True to defer this work.
        th : float, default=0.8
            Sequence-identity threshold used for weighted gap frequencies when
            ``fast=False``.

        Returns
        -------
        MultipleSequenceAlignment
            A new concatenated alignment.
        """
        if not alignments:
            raise ValueError("At least one MSA must be provided.")
        first, *remaining = alignments
        if not isinstance(first, MultipleSequenceAlignment):
            raise TypeError("Argument 1 is not a MultipleSequenceAlignment.")

        for i, other in enumerate(remaining, start=2):
            if not isinstance(other, MultipleSequenceAlignment):
                raise TypeError(f"Argument {i} is not a MultipleSequenceAlignment.")
            if other.L != first.L:
                raise ValueError(
                    f"Alignment length mismatch for argument {i}: "
                    f"got L={other.L}, expected L={first.L}."
                )
            if other.q != first.q or other.tokens != first.tokens:
                raise ValueError(f"Argument {i} does not use the same alphabet.")

        result = cls(
            np.concatenate(tuple(msa.headers for msa in alignments)),
            torch.cat(tuple(msa.enc for msa in alignments), dim=0).numpy(),
            setup=first.setup,
        )

        if all(
            msa._onehot_cache is not None
            and msa._onehot_device == first._onehot_device
            and msa._onehot_cache.dtype == first._onehot_cache.dtype
            for msa in alignments
        ):
            result._onehot_cache = torch.cat(
                tuple(msa._onehot_cache for msa in alignments), dim=0
            ).contiguous()
            result._onehot_device = first._onehot_device

        if not fast:
            result.compute_gap_frequency(th=th)
        return result

    # ------------------------- #
    # -- Clean the alignment -- #
    def remove_items(self, x, axis="columns", fast=False, th=0.8):
        """Remove rows or columns without constructing a one-hot copy.

        ``th`` is the sequence-identity threshold used when a weighted gap
        frequency is needed or when statistics are recomputed. By default the
        resulting MSA has current weights, ``Meff``, and gap frequencies; set
        ``fast=True`` only to defer those calculations while batching edits.
        """
        if axis in (1, "columns", "column", "cols", "col"):
            ax, name, size = 1, "columns", self.L
        elif axis in (0, "rows", "row"):
            ax, name, size = 0, "sequences", self.M
        else:
            raise ValueError("axis must be one of {'rows', 'columns', 0, 1}.")

        if isinstance(x, float):
            if not 0.0 <= x <= 1.0:
                raise ValueError("Gap frequency threshold must be within [0.0, 1.0].")
            gap_freq = (
                self.compute_gap_frequency(th=th).numpy()
                if ax == 1
                else (self.enc == 0).to(torch.float32).mean(dim=1).numpy()
            )
            idxs = np.where(gap_freq > x)[0]
            if idxs.size == 0:
                print(f"No {name} exceed threshold of {x:.2f} gap fraction.")
                return None
            print(f"Removing {name} with gap fraction > {x:.2f}: {idxs.tolist()}\n"
                  f"Number of {name} removed: {idxs.size}")
        else:
            idxs = np.asarray(x, dtype=int).ravel()
            if idxs.size == 0:
                print(f"No {name} indices provided.")
                return None
            if np.any(idxs >= size) or np.any(idxs < -size):
                raise IndexError(f"{name.capitalize()} index is out of range.")
            print(f"Removing {name} with indices: {idxs.tolist()}\n"
                  f"Number of {name} removed: {idxs.size}")

        # ``index_select`` makes one compact CPU replacement; there is no
        # NumPy/MPS round trip and no second giant one-hot alignment.
        keep = np.delete(np.arange(size), idxs)
        keep_tensor = torch.as_tensor(keep, dtype=torch.long)
        self.enc = self.enc.index_select(ax, keep_tensor).contiguous()
        if ax == 0:
            self.headers = np.delete(self.headers, idxs, axis=0)

        self.recompute_statistics(fast=fast, th=th)
        return idxs

    def remove_sequences_close_to_wildtype(
        self, wildtype, threshold, fast=False, th=0.8
    ):
        """Remove sequences too similar to one or more wildtype sequences.

        Similarity is measured as normalized Hamming distance, i.e. the
        fraction of aligned positions that differ.  A sequence is removed if
        its distance from *any* sequence in ``wildtype`` is less than or equal
        to ``threshold``.

        Parameters
        ----------
        wildtype : MultipleSequenceAlignment or ProteinSequence
            One or more wildtype sequences represented by an MSA, or one
            :class:`ProteinSequence`. For a protein, its aligned representation
            is used. The alignment length and alphabet must match this MSA.
        threshold : float
            Maximum allowed fraction of mismatches, in ``[0, 1]``.
        fast : bool, default=False
            If False, recompute the resulting MSA statistics. Set True to
            defer this work while batching edits.
        th : float, default=0.8
            Sequence-identity threshold used for weighted gap frequencies when
            statistics are recomputed.

        Returns
        -------
        numpy.ndarray
            Indices of removed sequences, or ``None`` if no sequence matches.
        """
        # Import locally so msa.py does not depend on protein.py at module load.
        from .protein import ProteinSequence

        if isinstance(wildtype, MultipleSequenceAlignment):
            wildtype_enc = wildtype.enc
            wildtype_L = wildtype.L
            wildtype_q = wildtype.q
            wildtype_tokens = wildtype.tokens
            if wildtype.M == 0:
                raise ValueError("wildtype must contain at least one sequence.")
        elif isinstance(wildtype, ProteinSequence):
            if wildtype.q != self.q or wildtype.tokens != self.tokens:
                raise ValueError("wildtype must use the same alphabet as this MSA.")
            aligned = wildtype.aligned.get_string()
            wildtype_enc = torch.tensor(
                encode_sequence(aligned, self.tokens),
                device="cpu",
                dtype=torch.int32,
            ).view(1, -1)
            wildtype_L = len(aligned)
            wildtype_q = wildtype.q
            wildtype_tokens = wildtype.tokens
        else:
            raise TypeError(
                "wildtype must be a MultipleSequenceAlignment or ProteinSequence."
            )

        if not isinstance(threshold, (float, int, np.integer, np.floating)):
            raise TypeError("threshold must be a number in [0.0, 1.0].")
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Distance threshold must be within [0.0, 1.0].")
        if wildtype_L != self.L:
            raise ValueError(
                f"Alignment length mismatch: got L={wildtype_L}, expected L={self.L}."
            )
        if wildtype_q != self.q or wildtype_tokens != self.tokens:
            raise ValueError("wildtype must use the same alphabet as this MSA.")

        # Compute the minimum distance to any wildtype without constructing a
        # one-hot tensor. The temporary tensor is at most M x N x L integers.
        distances = (self.enc[:, None, :] != wildtype_enc[None, :, :]).float().mean(dim=2)
        idxs = torch.where(distances.min(dim=1).values <= threshold)[0].numpy()
        if idxs.size == 0:
            print(f"No sequences are within distance threshold {threshold:.2f} of wildtype.")
            return None

        print(
            f"Removing sequences within distance threshold {threshold:.2f} of wildtype: "
            f"{idxs.tolist()}\nNumber of sequences removed: {idxs.size}"
        )
        keep = np.delete(np.arange(self.M), idxs)
        self.enc = self.enc.index_select(0, torch.as_tensor(keep, dtype=torch.long)).contiguous()
        self.headers = np.delete(self.headers, idxs, axis=0)
        self.recompute_statistics(fast=fast, th=th)
        return idxs

    # --------------------------------- #
    # -- Principal component analysis -- #
    def _to_onehot_for_projection(self, sequences, device=None) -> tuple[torch.Tensor, int]:
        """Convert input sequences to one-hot tensors on the chosen device."""
        target = self._resolve_device(device)
        if isinstance(sequences, MultipleSequenceAlignment):
            seq_oh = sequences.get_onehot(device=target)
            N = sequences.M
        elif isinstance(sequences, (torch.Tensor, np.ndarray)):
            tensor = torch.as_tensor(sequences, device=target)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.dim() == 2:
                if tensor.shape[-1] == self.q and tensor.dtype.is_floating_point:
                    seq_oh = tensor.unsqueeze(0).to(dtype=self.dtype)
                else:
                    seq_oh = one_hot(tensor.to(torch.int32), num_classes=self.q).to(self.dtype)
            elif tensor.dim() == 3:
                seq_oh = tensor.to(dtype=self.dtype)
            else:
                raise ValueError(
                    "Array input must be encoded with shape (L,) or (N, L), "
                    "or one-hot encoded with shape (L, q) or (N, L, q)."
                )
            N = seq_oh.shape[0]
        elif isinstance(sequences, str):
            seq_num = encode_sequence(sequences, self.tokens)
            seq_enc = torch.tensor(seq_num, device=target, dtype=torch.int32)
            seq_oh = one_hot(seq_enc.view(1, -1), num_classes=self.q).to(self.dtype)
            N = 1
        else:
            raise TypeError("sequences must be an MSA, a string, a torch tensor, or a numpy array.")

        if seq_oh.shape[1] != self.L:
            raise ValueError(f"Sequence length mismatch: got L={seq_oh.shape[1]}, expected L={self.L}.")
        if seq_oh.shape[2] != self.q:
            raise ValueError(f"Alphabet size mismatch: got q={seq_oh.shape[2]}, expected q={self.q}.")
        return seq_oh, N

    def project(self, seq, n_components=4, device=None):
        if n_components < 1:
            raise ValueError("n_components must be at least 1.")
        target = self._resolve_device(device)
        if self._V is None:
            self.compute_pca(n_components=n_components, device=target)
        elif self._V.device != target:
            raise ValueError("PCA is cached on another device; call clear_cache() and recompute it.")
        seq_oh, N = self._to_onehot_for_projection(seq, device=target)
        proj = ((seq_oh.reshape(N, -1) - self._pca_mean) @ self._V)[:, :n_components]
        return proj.cpu().numpy()

    def compute_pca_resample(self, n_components=4, device=None):
        if n_components < 1:
            raise ValueError("n_components must be at least 1.")
        target = self._resolve_device(device)
        if self.weights is None:
            self.compute_weights_cls(device=target)
        if self._V is not None and self._V.device != target:
            raise ValueError("PCA is cached on another device; call clear_cache() and recompute it.")

        # This large tensor is local: it is released when the method returns.
        msa_oh = self.get_onehot(device=target)
        if self._V is None:
            weights = self.weights.to(target, dtype=self.dtype)
            msa_w = resample_sequences(msa_oh, weights=weights, nextract=self.M)
            self._pca_mean = msa_w.reshape(self.M, -1).mean(0, keepdim=True)
            centered = msa_w.reshape(self.M, -1) - self._pca_mean
            self._V = torch.linalg.svd(centered, full_matrices=False)[2].T
        projection = ((msa_oh.reshape(self.M, -1) - self._pca_mean) @ self._V)[:, :n_components]
        return projection.cpu().numpy()

    def compute_pca(self, n_components=4, device=None):
        if n_components < 1:
            raise ValueError("n_components must be at least 1.")
        target = self._resolve_device(device)
        if self.weights is None:
            self.compute_weights_cls(device=target)
        if self._V is not None and self._V.device != target:
            raise ValueError("PCA is cached on another device; call clear_cache() and recompute it.")

        # The one-hot alignment is intentionally a local temporary tensor.
        msa_oh = self.get_onehot(device=target)
        X = msa_oh.reshape(self.M, -1)
        if self._V is None:
            weights = self.weights.to(target, dtype=self.dtype)
            weights = weights / weights.sum()
            self._pca_mean = (weights[:, None] * X).sum(dim=0, keepdim=True)
            centered = X - self._pca_mean
            weighted = centered * torch.sqrt(weights[:, None])
            self._V = torch.linalg.svd(weighted, full_matrices=False)[2].T
        projection = ((X - self._pca_mean) @ self._V)[:, :n_components]
        return projection.cpu().numpy()

    # ------------------------- #
    # -- Summary and writing -- #
    def write_to_file(self, path):
        write_fasta(path, self.headers, self.seqs, tokens=self.tokens)

    def summary(self, th=0.8):
        if self.gap_freq is None:
            self.compute_gap_frequency(th=th)
        print(f"Alignment name: {self.name}")
        print(f"Authorship: {self.author}")
        print(f"Sequence family: {self.family}")
        print("Number of amino acids per sequence:", self.L)
        print("Number of sequences:", self.M)
        print("Effective number of sequences:", self.Meff)
        print("Average gap fraction:", round(self.gap_freq.mean().item(), 2))
