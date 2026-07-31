from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.colors import LogNorm, Normalize
from scipy.stats import entropy as shannon_entropy
from scipy.stats import spearmanr

from adabmDCA.fasta import decode_sequence, encode_sequence
from adabmDCA.statmech import compute_energy


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
        model_positions = np.repeat([self.protein.u2a[pos] for pos in wt_positions], len(aa_list))
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
                raise Exception("Please compute the in-silico DMS scores by running `GENE.dms.compute(params)`")

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

