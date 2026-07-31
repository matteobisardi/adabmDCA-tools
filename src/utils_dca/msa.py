from __future__ import annotations

import numpy as np
import torch

from adabmDCA.fasta import compute_weights, encode_sequence, write_fasta
from adabmDCA.functional import one_hot
from adabmDCA.utils import resample_sequences

from .config import make_setup
from .fasta import import_from_fasta_keep_order


class MultipleSequenceAlignment:
    def __init__(self, headers, seqs, setup=None):
        
        if setup is None:
            setup = make_setup()

        # Get setup
        self._get_setup(setup)

        # Save sequences and headers
        self.headers = headers
        self.seqs = seqs
        self.enc = torch.tensor(self.seqs, device= self.device, dtype= torch.int32)
        self.onehot = one_hot(self.enc, num_classes= self.q).to(self.dtype)
        
        # Principal component analysis
        self._V = None
  
        # Compute basic statistics
        self.M = self.seqs.shape[0]
        self.L = self.seqs.shape[1]
        self.Meff = None

        # Setore other parameters
        self.gap_freq = None
        self.weights = None

        # Additional attributes
        self.family = ""
        self.author = ""
        self.name = ""
     
    def __len__(self):
        return self.M
    
    def __getitem__(self, idx: int):
        headers = self.headers[idx]
        sample = self.seqs[idx]
        return (headers,sample)


    # ---------------- #
    # -- Utilities -- #
    def _get_setup(self, setup):
        self.setup = setup
        self.device = self.setup["device"]
        self.dtype = self.setup["dtype"]
        self.tokens = self.setup["tokens"]
        self.q = self.setup["q"]



    # ---------------- #
    # -- Import MSA -- #
    @classmethod
    def from_path(cls, path, setup=None, remove_duplicates = False):
        if setup is None:
            setup = make_setup()
        tokens = setup["tokens"]
        headers,seqs = import_from_fasta_keep_order(path, tokens, filter_sequences=True, remove_duplicates=remove_duplicates)
        return cls(headers,seqs, setup)


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
        M = seqs.shape[0]
        headers = ["seq"+str(i) for i in range(1, M+1)]
        return cls(headers,seqs, setup)


    # ---------------- #
    # -- Statistics -- #
    # ---------------- #

    def compute_weights_cls(self, th = 0.8):
        self.weights = compute_weights(self.enc, th = th, device=self.device, dtype= self.dtype)
        self.Meff = int(self.weights.sum())
        return self.weights

    def compute_gap_frequency(self):
        self.gap_freq = self.onehot[:, :, 0].mean(dim=0).cpu()

    def recompute_statistics(self, fast = True, th = 0.8):
        # The alignment may have changed, so all derived state is stale.
        self.weights = None
        self.Meff = None
        self._V = None
        self._pca_mean = None
        self.M = self.seqs.shape[0]
        self.L = self.seqs.shape[1]
        self.compute_gap_frequency()
        if fast == False:
            self.compute_weights_cls(th = th)
        else:
            print("Remember to re-run compute_weights_cls()!")




    # ------------------------- #
    # -- Clean the alignment -- #
    # ------------------------- #
    def remove_items(self, x, axis="columns", fast=True):
        """
        Remove rows or columns, optionally using a gap-frequency threshold.

        x: sequence of indices or float in [0, 1] (gap-frequency cutoff).
        axis: 0/'rows' or 1/'columns'.
        fast: if True, skip recomputing weights and just print a reminder.
        """
        # Normalize axis and related names
        if axis in (1, "columns", "column", "cols", "col"):
            ax = 1
            name = "columns"
        elif axis in (0, "rows", "row"):
            ax = 0
            name = "sequences"
        else:
            raise ValueError("axis must be one of {'rows','columns',0,1}")

        # Determine which indices to remove
        if isinstance(x, float):
            if not (0.0 <= x <= 1.0):
                raise ValueError("Gap frequency threshold must be within [0.0, 1.0].")

            if ax == 1:
                if getattr(self, "gap_freq", None) is None:
                    self.compute_gap_frequency()
                gap_freq = np.asarray(self.gap_freq)
            else:
                # compute per-row gap freq: self.onehot shape (M,L,A), assume gap channel = 0
                gap_freq = self.onehot[:, :, 0].mean(dim=1).cpu().numpy()

            idxs = np.where(gap_freq > x)[0]
            if idxs.size == 0:
                print(f"No {name} exceed threshold of {x:.2f} gap fraction.")
                return
            print(f"Removing {name} with gap fraction > {x:.2f}: {idxs.tolist()}\n"
                f"Number of {name} removed: {idxs.size}")
        else:
            idxs = np.array(x, dtype=int).ravel()
            print(f"Removing {name} with indices: {idxs.tolist()}\n"
                  f"Number of {name} removed: {idxs.size}")
            if idxs.size == 0:
                print(f"No {name} indices provided.")
                return

        # Update sequences 
        self.seqs = np.delete(self.seqs, idxs, axis=ax)

        # optional headers handling
        if ax == 0 and getattr(self, "headers", None) is not None:
            self.headers = np.delete(self.headers,idxs, axis=0)

        # update encoding and onehot
        enc_cpu = np.delete(self.enc.cpu().numpy(), idxs, axis=ax)
        self.enc = torch.from_numpy(enc_cpu).to(self.device)    

        onehot_cpu = np.delete(self.onehot.cpu().numpy(), idxs, axis=ax)
        self.onehot = torch.from_numpy(onehot_cpu).to(self.device, dtype=self.dtype)

        # recompute statistics
        self.recompute_statistics(fast = fast)
        return idxs



    # --------------------------------- #
    # -- Princpal component analysis -- #
    # --------------------------------- #

    def _to_onehot_for_projection(self, sequences) -> torch.Tensor:
        """
        Internal helper: turn different sequence formats into one-hot on the
        same alphabet/length as this MSA.

        Accepted formats:
        - MultipleSequenceAlignment instance  -> uses its .onehot
        - torch.Tensor (N, L) of ints        -> encoded -> one_hot
        - torch.Tensor (N, L, q)             -> assumed one-hot
        - np.ndarray with same shapes as above

        Returns
        -------
        torch.Tensor
            One-hot tensor of shape (N, L, q) on self.device / self.dtype.
        """

        # Case 1: another MSA
        if isinstance(sequences, MultipleSequenceAlignment):
            seq_oh = sequences.onehot.to(self.device, dtype=self.dtype)
            N = sequences.M

        # Case 2: torch or numpy tensors
        elif isinstance(sequences, (torch.Tensor, np.ndarray)):
            tensor = torch.as_tensor(sequences, device=self.device)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)

            if tensor.dim() == 2:
                # Floating (L, q) input is interpreted as one single one-hot sequence.
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

        # Case 4: string
        elif isinstance(sequences, str):
            seq_num = encode_sequence(sequences, self.tokens)
            seq_enc = torch.tensor(seq_num, device= self.device, dtype= torch.int32)
            seq_oh = one_hot(seq_enc.view(1, -1), num_classes= self.q).to(self.dtype)
            N = 1
        else:
            raise TypeError(
                "sequences must be an MSA, a string, a torch tensor, or a numpy array."
            )

        # Basic sanity checks
        if seq_oh.shape[1] != self.L:
            raise ValueError(
                f"Sequence length mismatch: got L={seq_oh.shape[1]}, expected L={self.L}."
            )
        if seq_oh.shape[2] != self.q:
            raise ValueError(
                f"Alphabet size mismatch: got q={seq_oh.shape[2]}, expected q={self.q}."
            )

        return seq_oh, N

    def project(self, seq, n_components=4):

        if n_components < 1:
            raise ValueError("n_components must be at least 1.")

        if self._V is None:
            _ = self.compute_pca(n_components=n_components)
        scale_factor = 1 # self.L ** 0.5
        seq_oh, N = self._to_onehot_for_projection(seq)  # (N, L, q)
        proj = ( (seq_oh.view(N, -1) - self._pca_mean) @ self._V / scale_factor)[:, :n_components]
        return proj.cpu().numpy()

    def compute_pca_resample(self, n_components=4):
        if n_components < 1:
            raise ValueError("n_components must be at least 1.")
        M = self.M

        # Get weights
        if self.weights is None:
            weights = self.compute_weights_cls()
            # Set all weights to one
            #self.weights = torch.ones(M, device=self.device, dtype=self.dtype)

        # Extract a balanced sample of sequences
        if self._V is None:
            msa_w = resample_sequences(self.onehot, weights=self.weights, nextract=M)
            self._pca_mean = msa_w.view(M, -1).mean(0, keepdim=True)
            msa_w_centered = msa_w.view(M, -1) - self._pca_mean
            # SVD
            self._V = torch.linalg.svd(msa_w_centered, full_matrices=False)[2].T

        scale_factor = 1 #(self.L ** 0.5)
        # Project
        msa_proj = ( (self.onehot.view(M, -1) - self._pca_mean) @ self._V / scale_factor)[:, :n_components]

        return msa_proj.cpu().numpy()


    def compute_pca(self, n_components=4):
        if n_components < 1:
            raise ValueError("n_components must be at least 1.")
        M = self.M

        if self.weights is None:
            self.compute_weights_cls()

        if self._V is None:
            X = self.onehot.view(M, -1)

            weights = self.weights.view(-1).to(self.device, dtype=self.dtype)
            weights = weights / weights.sum()

            self._pca_mean = (weights[:, None] * X).sum(dim=0, keepdim=True)

            X_centered = X - self._pca_mean
            X_weighted = X_centered * torch.sqrt(weights[:, None])

            self._V = torch.linalg.svd(X_weighted, full_matrices=False)[2].T

        scale_factor = 1
        msa_proj = ((self.onehot.view(M, -1) - self._pca_mean) @ self._V / scale_factor)[:, :n_components]

        return msa_proj.cpu().numpy()


    # ------------------------- #
    # -- Summary and writing -- #
    def write_to_file(self, path):
        write_fasta(path, self.headers, self.seqs, tokens=self.tokens)

    def summary(self):
        print(f"Alignment name: {self.name}")
        print(f"Authorship: {self.author}")
        print(f"Sequence family: {self.family}")
        print(f"Number of amino acids per sequence:", self.L)
        print(f"Number of sequences:", self.M)
        print("Effective number of sequences:", self.Meff)
        if self.gap_freq is None:
            self.compute_gap_frequency()
        print("Average gap fraction:", round(self.gap_freq.mean().item(), 2))
