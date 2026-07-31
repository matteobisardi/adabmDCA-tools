from __future__ import annotations

# class MultipleSequenceAlignment

# packages
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

# adabmDCA
from adabmDCA.fasta import import_from_fasta, compute_weights, get_tokens, write_fasta, encode_sequence, decode_sequence
from adabmDCA.functional import one_hot
from adabmDCA.statmech import compute_energy
from adabmDCA.utils import resample_sequences

# class DeepMutationalScanning
from collections.abc import MutableSequence
from typing import Iterable, Optional, Tuple, Dict, List
from matplotlib.colors import Normalize, LogNorm
from scipy.stats import spearmanr
from scipy.stats import entropy as shannon_entropy 
from editdistance import eval as levenshtein_distance
from utils_dca.funcs import inverse_one_hot

class MultipleSequenceAlignment:
    def __init__(self, headers, seqs, setup):
        
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
    def from_path(cls, path, setup, remove_duplicates = True):
        tokens = setup["tokens"]
        headers,seqs = import_from_fasta(path, tokens, filter_sequences=True, remove_duplicates=remove_duplicates)
        return cls(headers,seqs, setup)


    @classmethod 
    def from_onehot(cls, msa_oh, setup):
        seqs = torch.argmax(msa_oh, dim=-1).cpu().numpy()
        M = seqs.shape[0]
        headers = ["seq"+str(i) for i in range(1, M+1)]
        return cls(headers,seqs, setup)


    # ---------------- #
    # -- Statistics -- #
    # ---------------- #

    def compute_weights_cls(self):
        self.weights = compute_weights(self.enc, device=self.device, dtype= self.dtype)
        self.Meff = int(self.weights.sum())
        return self.weights

    def compute_gap_frequency(self):
        self.gap_freq = self.onehot[:, :, 0].mean(dim=0).cpu()

    def recompute_statistics(self, fast = True):
        self.M = self.seqs.shape[0]
        self.L = self.seqs.shape[1]
        self.compute_gap_frequency()
        if fast == False:
            self.compute_weights_cls()
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

        # Case 2: torch.Tensor
        elif isinstance(sequences, torch.Tensor):
            if sequences.dim() == 3:
                # assume already one-hot
                seq_oh = sequences.to(self.device, dtype=self.dtype)
                N = sequences.shape[0] 
            else:
                raise ValueError("Torch tensor input must be 3D (one-hot). Single sequences must be of shape (1, L, q).")


        # Case 3: numpy array (to be done)

        # Case 4: string
        elif isinstance(sequences, str):
            seq_num = encode_sequence(sequences, self.tokens)
            seq_enc = torch.tensor(seq_num, device= self.device, dtype= torch.int32)
            seq_oh = one_hot(seq_enc.view(1, -1), num_classes= self.q).to(self.dtype)
            N = 1
        

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

    def project(self, seq):

        if self._V is None:
            _ = self.compute_pca()

        seq_oh, N = self._to_onehot_for_projection(seq)  # (N, L, q)
        proj = ( (seq_oh.view(N, -1) - self._pca_mean) @ self._V / (self.L ** 0.5))[:, :4]
        return proj.cpu().numpy()

    def compute_pca(self):
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

        scale_factor = (self.L ** 0.5)
        # Project
        msa_proj = ( (self.onehot.view(M, -1) - self._pca_mean) @ self._V / scale_factor)[:, :4]

        return msa_proj.cpu().numpy()



    # ------------------------- #
    # -- Summary and writing -- #
    def write_to_file(self, path):
        write_fasta(path, self.headers, self.seqs, numeric_input = True)

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

class ProteinSequence:
    def __init__(self, afa_sequence : str, setup, name : str = "GeneX" ):

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
        self.dms = DeepMutationalScanning(self)
        
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
                s2 = other.tolist()
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

class DeepMutationalScanning:
    def __init__(self, protein: ProteinSequence):
        
        # Store the protein object privately
        self._protein = protein
        self.L = self._protein.L
        self.L_unaligned = self._protein.L_unaligned

        # Save dms matrices
        self.model = None
        self.exp = None
        self.rho = None

        # Setup parameters
        self._get_setup(self._protein.setup)

    # Getters for protein
    @property
    def protein(self):
        return self._protein

    def _get_setup(self, setup):
        self.setup = setup
        self.device = self.setup["device"]
        self.dtype = self.setup["dtype"]
        self.tokens = self.setup["tokens"]
        self.q = self.setup["q"]

    # ---------- import the experimental dms scores ----------
    def from_standard_file(self, path: str) -> pd.DataFrame:
        """
        Load a standard DMS text file and build a dense experimental DataFrame.

        Fills missing (position, amino acid) pairs with NaN and checks that
        the DMS positions match the unaligned sequence length.
        """

        aa_list: List[str] = list(self.tokens[1:])

        # Read the file
        df = pd.read_csv(path, sep="\t")

        # Drop stop-codon rows
        df = df[df["mutated_aa"] != "*"].copy()

        # Ensure correct dtypes
        df["residue_num"] = df["residue_num"].astype(int)

        # Map residue_num -> wild-type amino acid (take the first observed)
        wt_map = df.groupby("residue_num")["wt_aa"].first()

        # Determine residue range (1 .. max)
        max_pos = df["residue_num"].max()
        min_pos = df["residue_num"].min()
        dms_length = max_pos - min_pos + 1
        if min_pos != 1 or dms_length != self.L_unaligned:
            raise ValueError(
                f"DMS positions must cover residues 1..{self.L_unaligned}, "
                f"but got {min_pos}..{max_pos}."
            )

        # Build complete index of all residue_num x amino_acid combinations
        full_index = pd.MultiIndex.from_product(
            [range(1, max_pos + 1), aa_list],
            names=["residue_num", "mutated_aa"]
        )

        # Reindex to create missing combinations with NaN values
        df_full = (
            df.set_index(["residue_num", "mutated_aa"])
                .reindex(full_index)
                .reset_index()
        )

        # Fill wt_aa using wt_map where possible; otherwise remains NaN
        df_full["wt_aa"] = df_full["wt_aa"].fillna(df_full["residue_num"].map(wt_map))

        # Rename columns
        df_full = df_full.rename(columns={
            "residue_num": "Position WT",
            "wt_aa": "WT AA",
            "mutated_aa": "Mutant AA",
            "log2_fitness_score": "Fitness"
        })

        # Ensure column order
        df_full = df_full[["Position WT", "WT AA", "Mutant AA", "Fitness"]]


        # Add a new column called "Position Model" 
        wt_positions = range(1, self.protein.L_unaligned + 1)
        model_positions = np.repeat([self.protein.u2a[pos] for pos in wt_positions], 20)
        df_full.insert(0, "Position Model", model_positions)

        self.df_exp = df_full.copy()
        self.from_experiment_dataframe(self.df_exp)

        return df_full

    # ---------- experimental data ----------
    def from_experiment_dataframe(self, df: pd.DataFrame, log_transform: bool = False ) -> None:
        """
        Store the experimental table and build a (q, L) fitness matrix.

        Uses 'Position WT', 'Mutant AA', and 'Fitness' columns; fills only
        aligned columns where a2u[j] is not None.
        """
        
        # Store the dataframe
        self.df_exp = df.copy()

        # # Check columns
        col_pos_wt = "Position WT"
        col_mut_aa = "Mutant AA"
        col_fitness = "Fitness"
        # for col in (col_pos_wt, col_mut_aa, col_fitness):
        #     print(col)
        #     if col not in self.df_exp.columns:
        #         print(self.df_exp.columns)
        #         raise ValueError(f"Experimental dataframe must contain column '{col}'")

        # Build empty matrix
        mat = np.full((self.q, self.L), np.nan, dtype=float)

        # For each aligned column, find its corresponding unaligned position
        tok2idx = {t: i for i, t in enumerate(self.tokens)}

        for j in range(1, self.L + 1):  # 1-based aligned
            u = self.protein.a2u[j]  # 1-based unaligned or None
            if u is None:
                continue  # a gap column in the alignment — keep column as NaN
            sub = self.df_exp[self.df_exp[col_pos_wt] == u]
            if sub.empty:
                continue
            # Fill by AA
            for aa, val in zip(sub[col_mut_aa].astype(str), sub[col_fitness].astype(float)):
                idx = tok2idx.get(aa)
                if idx is None:
                    continue
                mat[idx, j - 1] = val

        if log_transform:
            with np.errstate(divide="ignore"):
                mat = np.log(mat)

        self.exp = mat
        return self.exp

    # ---------- compute DMS using DCA (ΔE scores) ----------
    def compute(self,params) -> np.ndarray:
        """
        Compute ΔE = E(mutant) - E(WT) for all single mutants.

        Uses compute_energy on one-hot encodings and stores the (q, L) matrix
        in self.model.
        """

        q, L = self._protein.q, self._protein.L
        wt_idx = encode_sequence(self._protein._aligned, self.tokens)
        wt = torch.as_tensor(wt_idx, device=self.device, dtype=torch.long).unsqueeze(0)  # (1, L)
        wt_oh = F.one_hot(wt, num_classes=q).to(self.dtype)                               # (1, L, q)
        # energy fn may expect (N, L, q) or (N, q, L); keep your own convention
        E_wt = compute_energy(wt_oh, params).reshape(-1)                             # (1,) -> scalar

        mutants = []
        for i in range(L):
            for a in range(q):
                seq = wt.clone()
                seq[0, i] = a
                mutants.append(seq)
        mutants = torch.vstack(mutants)                                             # (q*L, L)
        mutants_oh = F.one_hot(mutants, num_classes=q).to(self.dtype)                    # (q*L, L, q)

        E_mut = compute_energy(mutants_oh, params).reshape(-1)                       # (q*L,)
        dE = (E_mut - E_wt).view(L, q).T.detach().cpu().numpy()                     # (q, L)

        self.model = dE
        return dE

    def _compute_dms_correlation(self):
        if self.exp is None or self.model is None:
            raise ValueError("Need both experimental and model matrices, please import or compute them.")
        exp = self.exp
        model = -self.model

        x = model.flatten()
        y = exp.flatten()
        rho, _ = spearmanr(x, y, nan_policy="omit")
        self.rho = rho

        x_mean = np.nanmean(model, axis=0)
        y_mean = np.nanmean(exp, axis=0)
        rho_mean, _ = spearmanr(x_mean, y_mean, nan_policy="omit")
        self.rho_mean = rho_mean

        return x, y, x_mean, y_mean

    # ---------- plot scatter between experimental and model scores ----------
    def plot_scatter( self, *, figsize: Tuple[float, float] = (6, 3), dpi: int = 200,  select_residues: Optional[Tuple[int, int]] = None):
        """
        Scatter plots of model vs experimental DMS scores.

        Left: per-position means; right: all amino-acid/position pairs.
        """
        #seq_range = (0, self.L)
        print(f"Plotting scatter for {self._protein.name}")
        if self.exp is None or self.model is None:
            raise ValueError("Need both experimental and model matrices, please import or compute them.")

        x, y, x_mean, y_mean = self._compute_dms_correlation()

        # Work with a restricted range
        if select_residues is None:
            select_residues = (0, self.L)
            rho = self.rho
            rho_mean = self.rho_mean

        else: 
            res1, res2 = select_residues
            exp = self.exp[:, res1:res2]
            model = -self.model[:, res1:res2]

            # Compute correlation for all points
            x = model.flatten()
            y = exp.flatten()
            rho, _ = spearmanr(x, y, nan_policy="omit")

            # Compute correlation for mean values
            x_mean = np.nanmean(model, axis=0)
            y_mean = np.nanmean(exp, axis=0)
            rho_mean, _ = spearmanr(x_mean, y_mean, nan_policy="omit")

        # -- Create figure --
        fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        fig.suptitle(self._protein.name, fontsize=14)

        # -- All points --
        ax[0].scatter(x, y, s=5, alpha=0.35, color = "salmon")
        ax[0].set_xlabel("Model")
        ax[0].set_ylabel("Experimental")
        ax[0].set_title(f"Spearman ρ = {np.round(rho, 2)}")
        self._clean_axes(ax[0])


        # -- Mean values --
        ax[1].scatter(x_mean, y_mean, s=8, alpha=0.5, color = "salmon")
        ax[1].set_xlabel("Model (mean)")
        ax[1].set_ylabel("Experimental (mean)")
        ax[1].set_title(f"Spearman ρ = {np.round(rho_mean, 2)}")
        self._clean_axes(ax[1])

        fig.tight_layout()
        return fig, ax

    # ---------- plot the heatmap of the dms scores ----------
    def plot_heatmap(self, 
        data_type: str,
        gap_index=0,
        summary_stat="mean",     # "entropy" or "mean"
        select_residues=None,    # None | (start, stop) | list/array[int] | array[bool]
        dpi=150,
        figsize=(32, 3.5),
        cmap=None,
        gene_name="GeneX",
        cmap_summary=None,
        xtick_step=5,
        log_floor=1e-4,
        pmin=10,                 # percentile lower (0..100)
        pmax=90,                 # percentile upper (0..100)
        summary_height=0.08,
        show_colorbar=True,
        ref_marker_kwargs=None,
        row_order="top",        ):
        """
        Plot a heatmap of model or experimental DMS scores.

        data_type must be 'model' or 'exp'. Returns (fig, axes_dict).
        """

        # ───────────────────────────────
        # 0) Get the datatype
        # ───────────────────────────────
        dms_matrix = getattr(self, data_type)
        gene_name = self.protein.name

        if data_type == "model":
            data_mode = "en"
            title = f"{gene_name} Deep Mutational Scan (model)"
            # For visualization reasons, change the sign of the scores to match DMS data
            if dms_matrix is not None:
                dms_matrix = -dms_matrix
            else:
                raise Exception("Please compute the in-silico DMS scores by running `GENE.dms.compute(params, setup)`")

        elif data_type == "exp":
            data_mode = "fitness"
            title = f"{gene_name} Deep Mutational Scan (experiment)"
            if dms_matrix is None:
                raise Exception("Please import the experimental DMS scores by running `GENE.dms.from_experiment(path)`")

        else:
            raise Exception("Beware! `data_type` can only be `model` or `exp`.")

        # ───────────────────────────────
        # 1) Normalize inputs (shapes & types)
        # ───────────────────────────────
        matrix = np.asarray(dms_matrix, dtype=float)
        Q, N = matrix.shape

        # Validate percentiles
        if not (0.0 <= float(pmin) < float(pmax) <= 100.0):
            raise ValueError("pmin/pmax must satisfy 0 <= pmin < pmax <= 100.")

        # A simple width heuristic so tiny regions don't look comically skinny:
        if figsize is None:
            xx = max(6, N // 8)
            yy = xx // 8
            figsize = (xx, yy)

        # ───────────────────────────────
        # 2) Prepare the reference (WT) slice, if present
        # ───────────────────────────────
        wt = self.protein._aligned
        wt_num = encode_sequence(wt, self.tokens)

        if wt_num is None:
            ref_slice = None
        else:
            ref_slice = np.asarray(wt_num)
            if ref_slice.dtype.kind in {"U", "S", "O"}:
                token_index = {tok: idx + 1 for idx, tok in enumerate(self.tokens)}
                ref_slice = np.array(
                    [token_index.get(str(tok), np.nan) for tok in ref_slice],
                    dtype=float,
                )
            else:
                ref_slice = ref_slice.astype(float)

        # ───────────────────────────────
        # 3) Resolve which columns (positions) to show
        # ───────────────────────────────
        if select_residues is None:
            col_idx = np.arange(N, dtype=int)

        elif isinstance(select_residues, tuple) and len(select_residues) == 2:
            start, stop = select_residues
            if start is None:
                start = 1
            if stop is None:
                stop = N

            if (start >= 1) and (stop >= 1):
                start0 = int(start) - 1
                stop0_excl = int(stop)
            else:
                start0 = int(start)
                stop0_excl = int(stop)

            start0 = max(0, min(N, start0))
            stop0_excl = max(0, min(N, stop0_excl))
            if stop0_excl < start0:
                raise ValueError("select_residues range is empty (stop < start).")

            col_idx = np.arange(start0, stop0_excl, dtype=int)

        else:
            arr = np.asarray(select_residues)
            if arr.dtype == bool:
                if arr.size != N:
                    raise ValueError("Boolean select_residues mask must have length N.")
                col_idx = np.where(arr)[0]
            else:
                if arr.ndim != 1:
                    raise ValueError("Explicit select_residues positions must be 1D.")
                if arr.size == 0:
                    raise ValueError("select_residues positions are empty.")
                arr = arr.astype(int)

                in_one_based = (arr.min() >= 1) and (arr.max() <= N)
                in_zero_based = (arr.min() >= 0) and (arr.max() <= (N - 1))
                if in_one_based and not in_zero_based:
                    arr = arr - 1
                elif in_zero_based and not in_one_based:
                    pass
                elif in_one_based and in_zero_based:
                    # Ambiguous but keep a consistent rule:
                    arr = arr - 1 if (arr.max() >= 1) else arr
                else:
                    raise ValueError("select_residues indices out of bounds or mixed 0- and 1-based.")

                _, first_idx = np.unique(arr, return_index=True)
                col_idx = arr[np.sort(first_idx)]

        data = matrix[:, col_idx]
        n_cols = data.shape[1]
        if ref_slice is not None:
            ref_slice = ref_slice[col_idx]

        display_positions = col_idx + 1

        # ───────────────────────────────
        # 4) Build AA row labels (and optionally flip the rows)
        # ───────────────────────────────
        try:
            aa_labels = [aa for aa in self.tokens]
            if len(aa_labels) != Q:
                raise ValueError
        except Exception:
            aa_labels = [f"{i+1}" for i in range(Q)]
        labels = aa_labels.copy()

        if row_order == "bottom":
            data = data[::-1, :]
            labels = labels[::-1]

        # ───────────────────────────────
        # 5) Mask columns that are "gaps" in the WT (if requested)
        # ───────────────────────────────
        gap_mask = np.zeros(n_cols, dtype=bool)
        if (ref_slice is not None) and (gap_index is not None):
            with np.errstate(invalid="ignore"):
                gap_mask = np.isclose(ref_slice, gap_index)
            if gap_mask.any():
                data[:, gap_mask] = np.nan
                ref_slice = ref_slice.copy()
                ref_slice[gap_mask] = np.nan

        # ───────────────────────────────
        # 6) Choose colormap + scaling that fit the data representation
        # ───────────────────────────────
        # Helper: percentile-based limits (robust to NaNs)
        def _percentile_limits(arr, plo, phi):
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return np.nan, np.nan
            lo = float(np.nanpercentile(finite, plo))
            hi = float(np.nanpercentile(finite, phi))
            if np.isclose(lo, hi):
                hi = lo + 1e-9
            return lo, hi

        # Colormap (defaulted per mode if not supplied)
        if cmap is None:
            if data_mode == "freq":
                cmap = plt.get_cmap("mako").copy()
            elif data_mode in {"fitness", "fit", "dms"}:
                cmap = plt.get_cmap("coolwarm").copy()
            elif data_mode in {"energy", "en"}:
                cmap = plt.get_cmap("Spectral_r").copy()
            else:
                raise ValueError("data_mode must be 'freq', 'fitness', or 'energy'")
        else:
            cmap = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap
        cmap.set_bad(color="#d9d9d9")

        # Compute percentile-defined normalization (always used)
        if data_mode == "freq":
            lo, hi = _percentile_limits(data, pmin, pmax)
            vmin_eff = max(log_floor, lo if np.isfinite(lo) else log_floor)
            vmax_eff = hi if np.isfinite(hi) else 1.0
            if not (vmax_eff > vmin_eff):
                vmax_eff = vmin_eff + 1e-9
            norm = LogNorm(vmin=vmin_eff, vmax=vmax_eff)
            cbar_label = "Frequency"
        elif data_mode in {"fitness", "fit", "dms"}:
            vmin_eff, vmax_eff = _percentile_limits(data, pmin, pmax)
            if not np.isfinite(vmin_eff):
                vmin_eff = -1.0
            if not np.isfinite(vmax_eff):
                vmax_eff = 1.0
            if np.isclose(vmin_eff, vmax_eff):
                vmax_eff = vmin_eff + 1e-9
            norm = Normalize(vmin=vmin_eff, vmax=vmax_eff)
            summary_norm = norm
            cbar_label = "Fitness"
        elif data_mode in {"energy", "en"}:
            vmin_eff, vmax_eff = _percentile_limits(data, pmin, pmax)
            if not np.isfinite(vmin_eff):
                vmin_eff = -1.0
            if not np.isfinite(vmax_eff):
                vmax_eff = 1.0
            if np.isclose(vmin_eff, vmax_eff):
                vmax_eff = vmin_eff + 1e-9
            norm = Normalize(vmin=vmin_eff, vmax=vmax_eff)
            summary_norm = norm
            cbar_label = "ΔE"

        # Keep the summary palette consistent unless explicitly overridden.
        if cmap_summary is None:
            cmap_summary = cmap

        # If summary_norm was not set in freq mode, default it to norm
        if data_mode == "freq":
            summary_norm = norm

        # ───────────────────────────────
        # 7) Compute the top summary band (one value per column)
        # ───────────────────────────────
        if summary_stat == "entropy":
            exp_data = np.exp(data)
            exp_data = np.nan_to_num(exp_data, nan=0.0)
            denom = np.sum(exp_data, axis=0, keepdims=True)
            denom[denom == 0] = 1.0
            probs = exp_data / denom
            summary_vals = np.apply_along_axis(shannon_entropy, 0, probs)
            summary_label = "Entropy"
        elif summary_stat == "mean":
            summary_vals = np.nanmean(data, axis=0)
            summary_label = "Mean"
        else:
            raise ValueError("summary_stat must be 'entropy' or 'mean'")

        # ───────────────────────────────
        # 8) Figure layout
        # ───────────────────────────────
        fig = plt.figure(figsize=figsize, dpi=dpi)
        summary_frac = float(np.clip(summary_height, 0.03, 0.5))
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=(1, 0.02 if show_colorbar else 0.01),
            height_ratios=(summary_frac, 1 - summary_frac),
            hspace=0.1,
            wspace=0.15,
        )
        ax_summary = fig.add_subplot(gs[0, 0])
        ax_heatmap = fig.add_subplot(gs[1, 0], sharex=ax_summary)
        cax = fig.add_subplot(gs[:, 1]) if show_colorbar else None

        # ───────────────────────────────
        # 9) Plot the main heatmap
        # ───────────────────────────────
        display_data = np.maximum(data, log_floor) if data_mode == "freq" else data
        masked = np.ma.masked_invalid(display_data)
        im = ax_heatmap.imshow(
            masked,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap=cmap,
            norm=norm,
        )

        ax_heatmap.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax_heatmap.set_yticks(np.arange(-0.5, Q, 1), minor=True)
        ax_heatmap.grid(which="minor", color="grey", linewidth=0.3)
        ax_heatmap.tick_params(which="minor", bottom=False, left=False)

        ax_heatmap.set_xlim(-0.5, n_cols - 0.5)
        ax_heatmap.set_ylim(-0.5, Q - 0.5)
        ax_heatmap.set_yticks(np.arange(Q))
        ax_heatmap.set_yticklabels(labels)
        ax_heatmap.set_title(title, fontsize=14, pad=60)
        ax_heatmap.set_xlabel("Residue position")
        ax_heatmap.set_ylabel("Amino acid")

        tick_positions = np.arange(0, n_cols, max(1, int(xtick_step)))
        ax_heatmap.set_xticks(tick_positions)
        ax_heatmap.set_xticklabels(display_positions[tick_positions], rotation=0)

        if show_colorbar:
            fig.colorbar(im, cax=cax, fraction=0.8, label=cbar_label)

        # ───────────────────────────────
        # 10) Plot the summary band on top
        # ───────────────────────────────
        summary_img = summary_vals[np.newaxis, :]
        finite = np.isfinite(summary_vals)
        if finite.any():
            ax_summary.imshow(summary_img, aspect="auto", cmap=cmap_summary, norm=summary_norm)
        else:
            ax_summary.imshow(summary_img, aspect="auto", cmap=cmap_summary)

        ax_summary.set_yticks([0])
        ax_summary.set_yticklabels([summary_label])
        ax_summary.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_summary.set_xlim(-0.5, n_cols - 0.5)
        ax_summary.tick_params(which="both", bottom=False, top=False)

        # ───────────────────────────────
        # 11) Overlay WT markers (tiny dots)
        # ───────────────────────────────
        if ref_slice is not None and np.isfinite(ref_slice).any():
            marker_cfg = {"marker": "o", "color": "black", "s": 14, "linewidth": 0, "alpha": 1}
            if ref_marker_kwargs:
                marker_cfg.update(ref_marker_kwargs)

            row_map = np.arange(Q) + 1
            if row_order == "bottom":
                row_map = row_map[::-1]

            rows = np.full(n_cols, np.nan)
            for j, ref_val in enumerate(ref_slice):
                if not np.isfinite(ref_val):
                    continue
                r0 = int(ref_val) - 1
                if 0 <= r0 < Q and not gap_mask[j]:
                    rows[j] = row_map[r0]

            valid = np.isfinite(rows)
            if valid.any():
                ax_heatmap.scatter(np.arange(n_cols)[valid], rows[valid], zorder=3, **marker_cfg)

        # ───────────────────────────────
        # 12) WT sequence letters above the summary bar
        # ───────────────────────────────
        if wt_num is not None:
            try:
                seq_str = decode_sequence(wt_num, tokens=self.tokens)
                seq_str = "".join(np.array(list(seq_str))[col_idx])
                for j, aa in enumerate(seq_str):
                    ax_summary.text(
                        j, 1, aa,
                        ha="center", va="bottom",
                        fontsize=6, fontweight="bold",
                        color="black",
                        transform=ax_summary.transData,
                    )
            except Exception as e:
                print(f"Could not decode WT sequence: {e}")

        # ───────────────────────────────
        # 13) Done
        # ───────────────────────────────
        fig.tight_layout()
        return fig, {"summary": ax_summary, "heatmap": ax_heatmap, "colorbar": cax}

    @staticmethod
    def _clean_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


# ---------- PPV (Contact Prediction) ----------
def compute_ppv_contacts(
    dca_scores: np.ndarray,
    ca_distances: np.ndarray,
    *,
    dist_cutoff: float = 8.0,
    min_seq_sep: int = 4,
    top_k: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute PPV (precision) for DCA contact prediction against real Ca-Ca distances.

    Only the upper triangle is used (i < j) and positions with |i-j| > min_seq_sep.

    Parameters
    ----------
    dca_scores : (L, L) array
        DCA score matrix (higher means more likely contact).
    ca_distances : (L, L) array
        Ca-Ca distance matrix in Angstrom.
    dist_cutoff : float
        Distance threshold to define a true contact.
    min_seq_sep : int
        Minimum sequence separation: only pairs with |i-j| > min_seq_sep are considered.
    top_k : int or None
        Number of top-scoring pairs to predict. If None, uses the number of true
        contacts among the considered pairs.

    Returns
    -------
    dict
        {"ppv": ..., "tp": ..., "fp": ..., "n_pred": ..., "n_true": ...}
    """
    scores = np.asarray(dca_scores, dtype=float)
    distances = np.asarray(ca_distances, dtype=float)

    if scores.shape != distances.shape or scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("dca_scores and ca_distances must be square matrices with the same shape.")

    L = scores.shape[0]
    # Upper triangle with sequence separation constraint (j - i > min_seq_sep)
    mask = np.triu(np.ones((L, L), dtype=bool), k=min_seq_sep + 1)

    scores_vec = scores[mask]
    true_vec = distances[mask] <= dist_cutoff

    n_true = int(true_vec.sum())
    if top_k is None:
        top_k = n_true

    top_k = int(top_k)
    if top_k < 0 or top_k > scores_vec.size:
        raise ValueError("top_k must be between 0 and the number of considered pairs.")

    if top_k == 0:
        return {"ppv": float("nan"), "tp": 0.0, "fp": 0.0, "n_pred": 0.0, "n_true": float(n_true)}

    top_idx = np.argsort(scores_vec)[::-1][:top_k]
    tp = int(true_vec[top_idx].sum())
    fp = int(top_k - tp)
    ppv = tp / top_k

    return {"ppv": float(ppv), "tp": float(tp), "fp": float(fp), "n_pred": float(top_k), "n_true": float(n_true)}


if __name__ == "__main__":
    print(
        "\nReminder from 'utils_dca.classes'. Define a dictionary like:\n"
        "setup = {'tokens': tokens, 'device': device, 'dtype': dtype, 'q': q}"
    )
