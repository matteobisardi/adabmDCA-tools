from os import PathLike
from pathlib import Path

import numpy as np
import torch

from adabmDCA.fasta import encode_sequence
from adabmDCA.functional import one_hot
from adabmDCA.statmech import compute_energy

from .config import make_setup
from .fasta import import_unaligned_fasta
from .msa import MultipleSequenceAlignment
from .protein import ProteinSequence


class SequencePath:
    """Directed mutational path between two aligned protein sequences.

    Each mutation replaces one residue of ``wildtype1`` with the residue found
    at the same position in ``wildtype2``. Mutation positions are 1-based.

    Parameters
    ----------
    wildtype1, wildtype2
        Starting and final aligned sequences. Each input can be a
        ``ProteinSequence``, a FASTA path, an encoded vector, or a string.
    params : dict
        DCA parameters containing ``bias`` and ``coupling_matrix``.
    algorithm : str, default="greedy"
        Path algorithm: ``"greedy"``, ``"flat"``, or ``"mean_energy"``.
    beta : float, default=1
        Inverse temperature used by the Monte Carlo algorithms.
    steps : int, default=10000
        Number of Monte Carlo steps.
    seed : int, optional
        Random seed for the Monte Carlo algorithms. If omitted, a random seed
        is generated.
    setup : dict, optional
        Alphabet and device setup used by the sequence objects.
    keep_history : bool, default=False
        If True, store the Monte Carlo objective after every sampling step.
    """

    def __init__(
        self,
        wildtype1,
        wildtype2,
        params,
        algorithm="greedy",
        beta=1,
        steps=10_000,
        seed=None,
        setup=None,
        keep_history=False,
    ):
        # Path definition and sampling options
        self.setup = make_setup() if setup is None else setup
        self.params = params
        self.algorithm = algorithm
        self.beta = beta
        self.steps = steps
        self.keep_history = keep_history
        if seed is None and algorithm in ("flat", "mean_energy"):
            seed = int(np.random.default_rng().integers(0, 2**32))
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Wildtypes and path results
        self.wildtype1 = self._wildtype_to_string(wildtype1)
        self.wildtype2 = self._wildtype_to_string(wildtype2)
        self.mutations = []
        self.energies = None
        self.score_history = [] if keep_history else None
        self.msa = None

        self.make_path()

    # --------------------------- #
    # -- Generate and use path -- #
    def make_path(self, seed=None):
        """Generate a path and update mutations, energies, and MSA.

        For a Monte Carlo algorithm, call without a seed to continue sampling
        from the current path. Pass a seed to restart from a new random path.
        A greedy path is deterministic and is not rebuilt.
        """
        path_exists = self.msa is not None
        if path_exists and self.algorithm == "greedy":
            print("The greedy path is deterministic and remains unchanged.")
            return self

        if seed is not None:
            self.seed = seed
            self._rng = np.random.default_rng(seed)

        start, end = self._prepare_wildtypes()
        different_positions = torch.where(start != end)[0].tolist()

        with torch.no_grad():
            order, energies = self._run_algorithm(
                start,
                end,
                different_positions,
                None if seed is not None or not path_exists else self.mutations,
            )

        self.mutations = [position + 1 for position in order]
        self.energies = energies.cpu().numpy()
        self.msa = self._build_msa()
        return self

    @classmethod
    def greedy(cls, wildtype1, wildtype2, params, setup=None):
        """Create a greedy minimum-energy path."""
        return cls(
            wildtype1,
            wildtype2,
            params,
            algorithm="greedy",
            setup=setup,
        )

    @classmethod
    def flat(
        cls,
        wildtype1,
        wildtype2,
        params,
        beta=1,
        steps=10_000,
        seed=None,
        setup=None,
        keep_history=False,
    ):
        """Create a path sampled with the flat-energy objective."""
        return cls(
            wildtype1,
            wildtype2,
            params,
            algorithm="flat",
            beta=beta,
            steps=steps,
            seed=seed,
            keep_history=keep_history,
            setup=setup,
        )

    @classmethod
    def mean_energy(
        cls,
        wildtype1,
        wildtype2,
        params,
        beta=1,
        steps=10_000,
        seed=None,
        setup=None,
        keep_history=False,
    ):
        """Create a path sampled with the mean-energy objective."""
        return cls(
            wildtype1,
            wildtype2,
            params,
            algorithm="mean_energy",
            beta=beta,
            steps=steps,
            seed=seed,
            keep_history=keep_history,
            setup=setup,
        )

    @classmethod
    def from_file(cls, path, params, setup=None):
        """Import an ordered mutational path from an aligned FASTA file.

        The first and last FASTA records are interpreted as the two wildtypes.
        Every intermediate record must add exactly one direct mutation toward
        the final wildtype.
        """

        if setup is None:
            setup = make_setup()

        headers, sequences = import_unaligned_fasta(path)
        if len(sequences) < 2:
            raise ValueError("A path FASTA must contain at least two sequences.")

        if any(len(sequence) != len(sequences[0]) for sequence in sequences):
            raise ValueError("All sequences in a path FASTA must have the same length.")
        tokens = set(setup["tokens"])
        if any(not set(sequence).issubset(tokens) for sequence in sequences):
            raise ValueError("The path FASTA contains an unknown token.")

        encoded = np.asarray(
            [encode_sequence(sequence, setup["tokens"]) for sequence in sequences]
        )

        # Recover the mutation order from consecutive FASTA records. Each step
        # must change one new position directly to its final residue.
        mutations = []
        for step in range(1, len(encoded)):
            changed = np.where(encoded[step - 1] != encoded[step])[0]
            if len(changed) != 1:
                raise ValueError(
                    "Each path step must differ from the previous sequence "
                    "at exactly one position."
                )

            position = changed[0]
            if encoded[step, position] != encoded[-1, position]:
                raise ValueError(
                    "Each path mutation must point directly toward wildtype2."
                )
            if position + 1 in mutations:
                raise ValueError("A path cannot mutate the same position twice.")

            mutations.append(position + 1)

        # Create the object from the imported path without generating a new
        # mutation order in __init__.
        sequence_path = cls.__new__(cls)
        sequence_path.setup = setup
        sequence_path.params = params
        sequence_path.algorithm = "imported"
        sequence_path.beta = None
        sequence_path.steps = None
        sequence_path.keep_history = False
        sequence_path.seed = None
        sequence_path._rng = np.random.default_rng()
        sequence_path.wildtype1 = sequences[0]
        sequence_path.wildtype2 = sequences[-1]
        sequence_path.mutations = mutations
        sequence_path.energies = None
        sequence_path.score_history = None
        sequence_path.msa = MultipleSequenceAlignment(headers, encoded, setup=setup)

        # Validate the endpoints against the model and score the imported path.
        sequence_path._prepare_wildtypes()
        with torch.no_grad():
            sequence_path.energies = (
                sequence_path._compute_energies(
                    sequence_path.msa.enc.to(
                        device=params["bias"].device,
                        dtype=torch.long,
                    )
                )
                .cpu()
                .numpy()
            )

        return sequence_path

    def __len__(self):
        return len(self.msa)

    def __getitem__(self, index):
        """Return one MSA row as ``(header, encoded_sequence)``."""
        return self.msa[index]

    def to_msa(self):
        """Return the path as a MultipleSequenceAlignment."""
        return self.msa

    def _build_msa(self):
        """Build the complete path as a MultipleSequenceAlignment."""
        start = encode_sequence(self.wildtype1, self.setup["tokens"])
        end = encode_sequence(self.wildtype2, self.setup["tokens"])

        current = start.copy()
        sequences = [current.copy()]

        for position in self.mutations:
            current = current.copy()
            current[position - 1] = end[position - 1]
            sequences.append(current)

        headers = ["wildtype1"]
        for step, position in enumerate(self.mutations[:-1], start=1):
            aa1 = self.wildtype1[position - 1]
            aa2 = self.wildtype2[position - 1]
            headers.append(f"step_{step}_{aa1}{position}{aa2}")
        if len(self.mutations) > 0:
            headers.append("wildtype2")

        return MultipleSequenceAlignment(
            headers,
            np.asarray(sequences),
            setup=self.setup,
        )

    def write_to_file(self, path):
        """Save the complete path as an aligned FASTA file."""
        self.msa.write_to_file(path)

    # ------------------------- #
    # -- Prepare input data -- #
    def _prepare_wildtypes(self):
        """Encode the two aligned wildtypes on the model device."""
        if len(self.wildtype1) != len(self.wildtype2):
            raise ValueError("The two aligned sequences must have the same length.")

        L = len(self.wildtype1)
        q = self.setup["q"]
        if self.params["bias"].shape != (L, q):
            raise ValueError("The DCA model does not match the sequences.")
        if self.params["coupling_matrix"].shape != (L, q, L, q):
            raise ValueError("The DCA model does not match the sequences.")

        device = self.params["bias"].device
        start = encode_sequence(self.wildtype1, self.setup["tokens"])
        end = encode_sequence(self.wildtype2, self.setup["tokens"])
        start = torch.tensor(start, dtype=torch.long, device=device)
        end = torch.tensor(end, dtype=torch.long, device=device)

        return start, end

    def _wildtype_to_string(self, wildtype):
        """Convert one supported wildtype input to an aligned string."""
        if isinstance(wildtype, ProteinSequence):
            return wildtype.aligned.get_string()

        if isinstance(wildtype, (list, tuple, np.ndarray, torch.Tensor)):
            encoded = torch.as_tensor(wildtype).detach().cpu()
            if encoded.ndim != 1:
                raise ValueError(
                    "An encoded wildtype must be a one-dimensional vector."
                )
            if not torch.equal(encoded, encoded.long()):
                raise ValueError("An encoded wildtype must contain integer values.")

            encoded = encoded.long()
            if torch.any(encoded < 0) or torch.any(encoded >= self.setup["q"]):
                raise ValueError("The encoded wildtype contains an unknown token.")

            return "".join(self.setup["tokens"][aa] for aa in encoded.tolist())

        if isinstance(wildtype, (str, PathLike)):
            path = Path(wildtype)
            is_path = isinstance(wildtype, PathLike)
            if not is_path:
                try:
                    is_path = path.is_file()
                except OSError:
                    is_path = False

            if is_path:
                _, sequences = import_unaligned_fasta(
                    path,
                    filter_sequences=False,
                    remove_duplicates=False,
                )
                if len(sequences) != 1:
                    raise ValueError("A wildtype FASTA must contain one sequence.")
                return sequences[0]

            return str(wildtype)

        raise TypeError(
            "A wildtype must be a ProteinSequence, FASTA path, encoded vector, "
            "or sequence string."
        )

    # --------------------- #
    # -- Path algorithms -- #
    def _run_algorithm(
        self,
        start,
        end,
        different_positions,
        initial_mutations=None,
    ):
        """Run the selected path algorithm."""
        if self.algorithm == "greedy":
            return self._greedy_path(start, end, different_positions)
        if self.algorithm == "flat":
            return self._monte_carlo_path(
                start,
                end,
                different_positions,
                self._flat_delta_score,
                self._flat_path_score,
                initial_mutations,
            )
        if self.algorithm == "mean_energy":
            return self._monte_carlo_path(
                start,
                end,
                different_positions,
                self._mean_energy_delta_score,
                self._mean_energy_path_score,
                initial_mutations,
            )

        raise ValueError("algorithm must be 'greedy', 'flat', or 'mean_energy'.")

    def _greedy_path(self, start, end, different_positions):
        # Begin from wildtype1 and calculate its full DCA energy once.
        current = start.clone()
        current_energy = self._compute_energies(current)[0]
        energies = [current_energy]
        order = []
        remaining = different_positions.copy()

        while len(remaining) > 0:
            # Generate every possible direct single-mutation among those available.
            candidate_sequences = current.repeat(len(remaining), 1)

            # Each row represents one of the remaining mutations.
            mutation_positions = torch.tensor(
                remaining,
                device=current.device,
            )

            # In every candidate, replace one wildtype1 residue with the
            # corresponding wildtype2 residue.
            candidate_sequences[
                torch.arange(len(remaining), device=current.device),
                mutation_positions,
            ] = end[mutation_positions]

            # Calculate the DCA energy change of all possible next mutations
            # together, relative to the current sequence.
            delta_energies = self._delta_energy(
                current.expand_as(candidate_sequences),
                candidate_sequences,
                mutation_positions[:, None],
            )

            # Select the mutation producing the lowest-energy intermediate.
            best = torch.argmin(delta_energies).item()
            position = remaining.pop(best)
            current = candidate_sequences[best]
            current_energy = current_energy + delta_energies[best]

            order.append(position)
            energies.append(current_energy)

        return order, torch.stack(energies)

    def _monte_carlo_path(
        self,
        start,
        end,
        different_positions,
        delta_score_function,
        path_score_function,
        initial_mutations=None,
    ):
        # Start a new chain from a random path. When make_path() is called on
        # an existing object without a seed, continue from its current order.
        if initial_mutations is None:
            order = self._rng.permutation(different_positions).tolist()
        else:
            order = [position - 1 for position in initial_mutations]
        path, energies = self._build_path(start, end, order)

        if self.keep_history:
            current_score = path_score_function(energies).item()
            if initial_mutations is None or self.score_history is None:
                self.score_history = [current_score]

        if len(order) < 2:
            return order, energies

        for _ in range(self.steps):
            # Select any two mutation times, not necessarily adjacent.
            left, right = sorted(
                self._rng.choice(len(order), 2, replace=False)
            )

            # Propose a new path by exchanging the times of the two mutations.
            new_order = order.copy()
            new_order[left], new_order[right] = new_order[right], new_order[left]

            # The prefix before `left` is unchanged. Rebuild every intermediate
            # from `left + 1` through `right` using the proposed mutation order.
            # The state after `right` is unchanged because both exchanged
            # mutations have occurred by then.
            new_path = []
            sequence = path[left].clone()

            for time in range(left, right):
                position = new_order[time]
                sequence = sequence.clone()
                sequence[position] = end[position]
                new_path.append(sequence.clone())

            new_path = torch.stack(new_path)

            # Corresponding old and proposed intermediates differ only at the
            # two exchanged residue positions. Calculate all their energy
            # changes together without rescoring the complete sequences.
            old_path = path[left + 1 : right + 1]
            delta_energies = self._delta_energy(
                old_path,
                new_path,
                torch.tensor(
                    [order[left], order[right]],
                    device=path.device,
                )[None, :].expand(len(new_path), -1),
            )
            new_energies = energies[left + 1 : right + 1] + delta_energies

            # The sampler is independent of the path objective. The selected
            # score function evaluates only the part changed by this proposal.
            delta_score = delta_score_function(
                energies,
                new_energies,
                left,
                right,
            )

            # Metropolis acceptance: always accept an improvement; otherwise
            # accept with probability exp(-beta * delta_score).
            delta_score = delta_score.item()
            accept = delta_score <= 0
            if not accept:
                accept = self._rng.random() < np.exp(-self.beta * delta_score)

            if accept:
                # Keep the proposed order, intermediate sequences, and their
                # energies synchronized.
                order = new_order
                path[left + 1 : right + 1] = new_path
                energies[left + 1 : right + 1] = new_energies
                if self.keep_history:
                    current_score += delta_score

            if self.keep_history:
                self.score_history.append(current_score)

        return order, energies

    def _flat_path_score(self, energies):
        """Sum of the squared energy jumps along the complete path."""
        return torch.sum(torch.diff(energies) ** 2)

    def _flat_delta_score(self, energies, new_energies, left, right):
        """Change in the squared energy jumps along the path."""
        # Include the unchanged sequence immediately before and after the
        # rebuilt segment because the jumps at both boundaries also change.
        old_window = energies[left : right + 2]
        new_window = torch.cat(
            (
                energies[left : left + 1],
                new_energies,
                energies[right + 1 : right + 2],
            )
        )
        return torch.sum(torch.diff(new_window) ** 2) - torch.sum(
            torch.diff(old_window) ** 2
        )

    def _mean_energy_path_score(self, energies):
        """Mean DCA energy of the complete path."""
        return torch.mean(energies)

    def _mean_energy_delta_score(self, energies, new_energies, left, right):
        """Change in the mean DCA energy of the complete path."""
        # Endpoints and all sequences outside the rebuilt segment are
        # unchanged, so they cancel when comparing the two path means.
        old_energies = energies[left + 1 : right + 1]
        return torch.sum(new_energies - old_energies) / len(energies)

    def _build_path(self, start, end, order):
        # Apply the ordered mutations one at a time, retaining both endpoints.
        current = start.clone()
        path = [current.clone()]

        for position in order:
            current = current.clone()
            current[position] = end[position]
            path.append(current.clone())

        path = torch.stack(path)

        # This is the initial energy evaluation for a Monte Carlo run. Score
        # the complete path in one batched call to adabmDCA.compute_energy().
        energies = self._compute_energies(path)
        return path, energies

    # ------------------------- #
    # -- Energy calculations -- #
    def _compute_energies(self, sequences):
        """Compute DCA energies with adabmDCA using temporary one-hot tensors."""
        # Treat a single sequence as a batch containing one sequence.
        if sequences.ndim == 1:
            sequences = sequences.unsqueeze(0)

        # adabmDCA computes energies from one-hot sequences. This representation
        # is temporary and is discarded after the energy calculation.
        sequences_oh = one_hot(
            sequences,
            num_classes=self.setup["q"],
        ).to(self.params["bias"].dtype)

        return compute_energy(sequences_oh, self.params)

    def _delta_energy(self, old_sequences, new_sequences, changed_positions):
        """Compute exact energy changes from only the affected couplings.

        ``changed_positions`` has shape ``(batch_size, n_changes)`` and lists
        the sequence positions that changed in each old/new sequence pair.
        """
        # DCA energy is minus the model score. Terms that do not involve a
        # changed position are identical in the two sequences and cancel.
        return self._affected_score(
            old_sequences,
            changed_positions,
        ) - self._affected_score(
            new_sequences,
            changed_positions,
        )

    def _affected_score(self, sequences, changed_positions):
        """Calculate model-score terms involving the changed positions."""
        bias = self.params["bias"]
        coupling = self.params["coupling_matrix"]

        changed_amino_acids = sequences.gather(1, changed_positions)
        changed = changed_positions[:, :, None]
        changed_aa = changed_amino_acids[:, :, None]
        all_positions = torch.arange(
            sequences.shape[1],
            device=sequences.device,
        )[None, None, :]
        all_aa = sequences[:, None, :]

        # Add coupling rows and columns involving a changed position. Their
        # intersection is subtracted because it occurs in both sums.
        coupling_score = (
            coupling[changed, changed_aa, all_positions, all_aa].sum(dim=(1, 2))
            + coupling[all_positions, all_aa, changed, changed_aa].sum(dim=(1, 2))
            - coupling[
                changed,
                changed_aa,
                changed.transpose(1, 2),
                changed_aa.transpose(1, 2),
            ].sum(dim=(1, 2))
        )

        return bias[changed_positions, changed_amino_acids].sum(dim=1) + (
            0.5 * coupling_score
        )

class SequencePathFast(SequencePath):
    """SequencePath using precomputed single and pair mutation effects.

    The public interface is the same as :class:`SequencePath`. For ``flat``
    and ``mean_energy`` paths, DCA terms are evaluated once before sampling.
    Monte Carlo then operates only on the mutation order and its energy steps.
    """

    def __init__(
        self,
        wildtype1,
        wildtype2,
        params,
        algorithm="greedy",
        beta=1,
        steps=10_000,
        seed=None,
        setup=None,
        keep_history=False,
    ):
        # Exact binary representation of the DCA energy landscape. These
        # attributes remain None for a greedy or imported path.
        self.reference_energy = None
        self.single_effects = None
        self.pair_effects = None
        self._effect_positions = None

        super().__init__(
            wildtype1,
            wildtype2,
            params,
            algorithm=algorithm,
            beta=beta,
            steps=steps,
            seed=seed,
            setup=setup,
            keep_history=keep_history,
        )

    @classmethod
    def from_file(cls, path, params, setup=None):
        """Import an ordered path using the SequencePath FASTA checks."""
        sequence_path = super().from_file(path, params, setup=setup)
        sequence_path.reference_energy = None
        sequence_path.single_effects = None
        sequence_path.pair_effects = None
        sequence_path._effect_positions = None
        return sequence_path

    def _run_algorithm(
        self,
        start,
        end,
        different_positions,
        initial_mutations=None,
    ):
        """Run greedy normally or use the precomputed Monte Carlo sampler."""
        if self.algorithm == "greedy":
            return self._greedy_path(start, end, different_positions)
        if self.algorithm == "flat":
            return self._fast_monte_carlo_path(
                start,
                end,
                different_positions,
                self._flat_step_delta_score,
                self._flat_step_score,
                initial_mutations,
            )
        if self.algorithm == "mean_energy":
            return self._fast_monte_carlo_path(
                start,
                end,
                different_positions,
                self._mean_step_delta_score,
                self._mean_step_score,
                initial_mutations,
            )

        raise ValueError("algorithm must be 'greedy', 'flat', or 'mean_energy'.")

    def _prepare_effects(self, start, end, different_positions):
        """Precompute the exact energy effect of every mutation pair."""
        if self.single_effects is not None:
            return

        self._effect_positions = different_positions.copy()
        self.reference_energy = self._compute_energies(start)[0].detach().cpu()

        if len(different_positions) == 0:
            self.single_effects = torch.empty(
                0,
                dtype=self.reference_energy.dtype,
            )
            self.pair_effects = torch.empty(
                (0, 0),
                dtype=self.reference_energy.dtype,
            )
            return

        mutation_positions = torch.tensor(
            different_positions,
            device=start.device,
        )

        # Calculate every single-mutation effect relative to wildtype1 in one
        # batch. This includes fields, fixed residues, and diagonal couplings.
        single_mutants = start.repeat(len(different_positions), 1)
        single_mutants[
            torch.arange(len(different_positions), device=start.device),
            mutation_positions,
        ] = end[mutation_positions]
        self.single_effects = self._delta_energy(
            start.expand_as(single_mutants),
            single_mutants,
            mutation_positions[:, None],
        ).detach().cpu()

        # Restrict each mutable position to its wildtype1/wildtype2 residues.
        # The second difference of Jij is the epistatic effect of each pair.
        old_amino_acids = start[mutation_positions]
        new_amino_acids = end[mutation_positions]
        coupling = self.params["coupling_matrix"]
        direct_pair_effects = (
            coupling[
                mutation_positions[:, None],
                new_amino_acids[:, None],
                mutation_positions[None, :],
                new_amino_acids[None, :],
            ]
            - coupling[
                mutation_positions[:, None],
                new_amino_acids[:, None],
                mutation_positions[None, :],
                old_amino_acids[None, :],
            ]
            - coupling[
                mutation_positions[:, None],
                old_amino_acids[:, None],
                mutation_positions[None, :],
                new_amino_acids[None, :],
            ]
            + coupling[
                mutation_positions[:, None],
                old_amino_acids[:, None],
                mutation_positions[None, :],
                old_amino_acids[None, :],
            ]
        )

        # compute_energy includes both Jij and Jji with a factor of one half.
        # Combining both directions also supports non-symmetric couplings.
        self.pair_effects = (
            -0.5 * (direct_pair_effects + direct_pair_effects.T)
        ).detach().cpu()
        self.pair_effects.fill_diagonal_(0)

    def _fast_monte_carlo_path(
        self,
        start,
        end,
        different_positions,
        delta_score_function,
        path_score_function,
        initial_mutations=None,
    ):
        self._prepare_effects(start, end, different_positions)

        # Internally the order indexes the precomputed mutation effects. The
        # public mutation positions are restored before returning.
        if initial_mutations is None:
            order = self._rng.permutation(len(different_positions)).tolist()
        else:
            position_to_index = {
                position: index
                for index, position in enumerate(different_positions)
            }
            order = [
                position_to_index[position - 1]
                for position in initial_mutations
            ]

        energy_steps = self._ordered_energy_steps(order)

        if self.keep_history:
            current_score = path_score_function(energy_steps).item()
            if initial_mutations is None or self.score_history is None:
                self.score_history = [current_score]

        if len(order) >= 2:
            for _ in range(self.steps):
                left, right = sorted(
                    self._rng.choice(len(order), 2, replace=False)
                )
                mutation_left = order[left]
                mutation_right = order[right]
                middle = torch.tensor(order[left + 1 : right], dtype=torch.long)

                # Swapping two mutation times changes only the energy steps in
                # this interval. Update them directly from the pair effects.
                new_energy_steps = energy_steps[left : right + 1].clone()
                new_energy_steps[0] = energy_steps[right] - self.pair_effects[
                    mutation_right,
                    mutation_left,
                ]
                new_energy_steps[-1] = energy_steps[left] + self.pair_effects[
                    mutation_left,
                    mutation_right,
                ]

                if len(middle) > 0:
                    new_energy_steps[0] -= self.pair_effects[
                        mutation_right,
                        middle,
                    ].sum()
                    new_energy_steps[-1] += self.pair_effects[
                        mutation_left,
                        middle,
                    ].sum()
                    new_energy_steps[1:-1] += (
                        self.pair_effects[middle, mutation_right]
                        - self.pair_effects[middle, mutation_left]
                    )

                delta_score = delta_score_function(
                    energy_steps,
                    new_energy_steps,
                    left,
                    right,
                ).item()

                accept = delta_score <= 0
                if not accept:
                    accept = self._rng.random() < np.exp(
                        -self.beta * delta_score
                    )

                if accept:
                    order[left], order[right] = order[right], order[left]
                    energy_steps[left : right + 1] = new_energy_steps
                    if self.keep_history:
                        current_score += delta_score

                if self.keep_history:
                    self.score_history.append(current_score)

        energies = torch.cat(
            (
                self.reference_energy[None],
                self.reference_energy + torch.cumsum(energy_steps, dim=0),
            )
        )
        return [different_positions[index] for index in order], energies

    def _ordered_energy_steps(self, order):
        """Calculate the energy added at each mutation time."""
        if len(order) == 0:
            return self.single_effects.clone()

        order = torch.tensor(order, dtype=torch.long)
        ordered_pairs = self.pair_effects[
            order[:, None],
            order[None, :],
        ]
        return self.single_effects[order] + torch.tril(
            ordered_pairs,
            diagonal=-1,
        ).sum(dim=1)

    def _flat_step_score(self, energy_steps):
        """Flat-path objective from mutation energy increments."""
        return torch.sum(energy_steps**2)

    def _flat_step_delta_score(
        self,
        energy_steps,
        new_energy_steps,
        left,
        right,
    ):
        """Change in the flat-path objective after a proposed swap."""
        return torch.sum(new_energy_steps**2) - torch.sum(
            energy_steps[left : right + 1] ** 2
        )

    def _mean_step_score(self, energy_steps):
        """Mean energy of every sequence along the path."""
        return torch.mean(
            torch.cat(
                (
                    self.reference_energy[None],
                    self.reference_energy + torch.cumsum(energy_steps, dim=0),
                )
            )
        )

    def _mean_step_delta_score(
        self,
        energy_steps,
        new_energy_steps,
        left,
        right,
    ):
        """Change in mean path energy after a proposed swap."""
        changed_steps = (
            new_energy_steps - energy_steps[left : right + 1]
        )
        return torch.sum(torch.cumsum(changed_steps, dim=0)) / (
            len(energy_steps) + 1
        )
