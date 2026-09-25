from os import PathLike
from pathlib import Path

import numpy as np
import torch

from adabmDCA.fasta import encode_sequence
from adabmDCA.functional import one_hot
from adabmDCA.statmech import compute_energy

from .config import make_setup
from .fasta import import_unaligned_fasta
from .metrics import compute_conditional_entropies, compute_conditional_logits
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
        Path algorithm: ``"random"``, ``"greedy"``, ``"flat"``, or
        ``"mean"``.
    beta : float, default=1
        Inverse temperature used by the Monte Carlo algorithms.
    steps : int, default=10000
        Number of Monte Carlo steps.
    seed : int, optional
        Random seed for random and Monte Carlo paths. If omitted, a random
        seed is generated.
    setup : dict, optional
        Alphabet and device setup used by the sequence objects.
    keep_history : bool, default=False
        If True, store the path objective after every Monte Carlo step in
        ``score_history``.
    use_free_energy : bool, default=False
        If True, use ``energy - slope * conditional_entropy`` as the path score.
    slope : float, default=1
        Weight of the summed Potts conditional entropy in the free energy.

    The selected path score is available as ``scores``. The returned path
    always includes ``energies``, ``entropies``, and ``free_energies``.
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
        use_free_energy=False,
        slope=1,
    ):
        # Path definition and sampling options
        self.setup = make_setup() if setup is None else setup
        self.params = params
        self.algorithm = "mean" if algorithm == "mean_energy" else algorithm
        self.beta = beta
        self.steps = steps
        self.keep_history = keep_history
        self.use_free_energy = use_free_energy
        self.slope = slope
        self._reference_conditional_logits = None
        self._conditional_effects = None
        self._conditional_positions = None
        if seed is None and self.algorithm in ("random", "flat", "mean"):
            seed = int(np.random.default_rng().integers(0, 2**32))
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Wildtypes and path results
        self.wildtype1 = self._wildtype_to_string(wildtype1)
        self.wildtype2 = self._wildtype_to_string(wildtype2)
        self.mutations = []
        self.energies = None
        self.entropies = None
        self.free_energies = None
        self.scores = None
        self.score_history = [] if keep_history else None
        self.msa = None

        self.make_path()

    # --------------------------- #
    # -- Generate and use path -- #
    def make_path(self, seed=None):
        """Generate a path and update mutations, scores, and MSA.

        A random path is redrawn on every call. A Monte Carlo path continues
        sampling from its current order. Pass a seed to restart either path
        reproducibly. A greedy path is deterministic and is not rebuilt.
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
            order, scores = self._run_algorithm(
                start,
                end,
                different_positions,
                None if seed is not None or not path_exists else self.mutations,
            )

        self.mutations = [position + 1 for position in order]
        self.msa = self._build_msa()
        encoded_path = torch.as_tensor(
            self.msa.enc,
            dtype=torch.long,
            device=self.params["bias"].device,
        )
        with torch.no_grad():
            self.energies = self._compute_energies(encoded_path).cpu().numpy()
            self.entropies = self._compute_entropies(encoded_path).cpu().numpy()
        self.free_energies = self.energies - self.slope * self.entropies
        self.scores = (
            self.free_energies.copy() if self.use_free_energy else self.energies.copy()
        )
        return self

    @classmethod
    def greedy(
        cls,
        wildtype1,
        wildtype2,
        params,
        setup=None,
        use_free_energy=False,
        slope=1,
    ):
        """Create a greedy path that minimizes the selected score at each step."""
        return cls(
            wildtype1,
            wildtype2,
            params,
            algorithm="greedy",
            setup=setup,
            use_free_energy=use_free_energy,
            slope=slope,
        )

    @classmethod
    def random(
        cls,
        wildtype1,
        wildtype2,
        params,
        seed=None,
        setup=None,
        use_free_energy=False,
        slope=1,
    ):
        """Create an unoptimized path with a random mutation order."""
        return cls(
            wildtype1,
            wildtype2,
            params,
            algorithm="random",
            seed=seed,
            setup=setup,
            use_free_energy=use_free_energy,
            slope=slope,
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
        use_free_energy=False,
        slope=1,
    ):
        """Sample a path that minimizes squared changes in the selected score."""
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
            use_free_energy=use_free_energy,
            slope=slope,
        )

    @classmethod
    def mean(
        cls,
        wildtype1,
        wildtype2,
        params,
        beta=1,
        steps=10_000,
        seed=None,
        setup=None,
        keep_history=False,
        use_free_energy=False,
        slope=1,
    ):
        """Create a path sampled to minimize the mean selected score."""
        return cls(
            wildtype1,
            wildtype2,
            params,
            algorithm="mean",
            beta=beta,
            steps=steps,
            seed=seed,
            keep_history=keep_history,
            setup=setup,
            use_free_energy=use_free_energy,
            slope=slope,
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
        use_free_energy=False,
        slope=1,
    ):
        """Compatibility alias for :meth:`mean`."""
        return cls.mean(
            wildtype1,
            wildtype2,
            params,
            beta=beta,
            steps=steps,
            seed=seed,
            setup=setup,
            keep_history=keep_history,
            use_free_energy=use_free_energy,
            slope=slope,
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
        sequence_path.use_free_energy = False
        sequence_path.slope = 1
        sequence_path.keep_history = False
        sequence_path.seed = None
        sequence_path._rng = np.random.default_rng()
        sequence_path.wildtype1 = sequences[0]
        sequence_path.wildtype2 = sequences[-1]
        sequence_path.mutations = mutations
        sequence_path.energies = None
        sequence_path.entropies = None
        sequence_path.free_energies = None
        sequence_path.scores = None
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
            sequence_path.entropies = (
                sequence_path._compute_entropies(
                    sequence_path.msa.enc.to(
                        device=params["bias"].device,
                        dtype=torch.long,
                    )
                )
                .cpu()
                .numpy()
            )
            sequence_path.free_energies = (
                sequence_path.energies
                - sequence_path.slope * sequence_path.entropies
            )
            sequence_path.scores = sequence_path.energies.copy()

        return sequence_path

    def __len__(self):
        return len(self.msa)

    def __getitem__(self, index):
        """Return one MSA row as ``(header, encoded_sequence)``."""
        return self.msa[index]

    def to_msa(self):
        """Return the path as a MultipleSequenceAlignment."""
        return self.msa

    def distance_k(self, another_path):
        """Return the Kendall distance between two mutation orders.

        This is the fraction of mutation pairs whose relative order is
        reversed between the paths. Both paths must contain the same mutation
        positions; the result ranges from zero (same order) to one (reverse
        order). Paths with fewer than two mutations have distance zero.
        """
        other_mutations = getattr(another_path, "mutations", None)
        if other_mutations is None:
            raise TypeError("another_path must be a SequencePath.")
        if len(self.mutations) != len(other_mutations) or set(self.mutations) != set(
            other_mutations
        ):
            raise ValueError("Both paths must contain the same mutation positions.")

        other_order = {position: rank for rank, position in enumerate(other_mutations)}
        other_ranks = [other_order[position] for position in self.mutations]
        inversions = sum(
            other_ranks[i] > other_ranks[j]
            for i in range(len(other_ranks))
            for j in range(i + 1, len(other_ranks))
        )
        n_pairs = len(other_ranks) * (len(other_ranks) - 1) // 2
        return inversions / n_pairs if n_pairs else 0.0

    def distance_introduction(self, another_path):
        """Return per-mutation differences in introduction time.

        For each mutation, the returned value is its 1-based introduction
        step in this path minus its 1-based introduction step in
        ``another_path``. Values are ordered by 1-based alignment position
        and can be negative. Both paths must contain the same mutations.
        """
        other_mutations = getattr(another_path, "mutations", None)
        if other_mutations is None:
            raise TypeError("another_path must be a SequencePath.")
        if len(self.mutations) != len(other_mutations) or set(self.mutations) != set(
            other_mutations
        ):
            raise ValueError("Both paths must contain the same mutation positions.")

        my_times = {
            position: time for time, position in enumerate(self.mutations, start=1)
        }
        other_times = {
            position: time for time, position in enumerate(other_mutations, start=1)
        }
        return np.asarray(
            [my_times[position] - other_times[position] for position in sorted(my_times)],
            dtype=int,
        )

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
        if self.algorithm == "random":
            return self._random_path(start, end, different_positions)
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
        if self.algorithm == "mean":
            return self._monte_carlo_path(
                start,
                end,
                different_positions,
                self._mean_delta_score,
                self._mean_path_score,
                initial_mutations,
            )

        raise ValueError("algorithm must be 'random', 'greedy', 'flat', or 'mean'.")

    def _random_path(self, start, end, different_positions):
        """Draw one uniformly random ordering of the directed mutations."""
        order = self._rng.permutation(different_positions).tolist()
        _, scores = self._build_path(start, end, order)
        return order, scores

    def _greedy_path(self, start, end, different_positions):
        # Begin from wildtype1 and calculate its selected path score once.
        current = start.clone()
        if self.use_free_energy:
            self._prepare_conditional_effects(start, end, different_positions)
            current_logits = self._reference_conditional_logits.clone()
            current_score = self._scores_from_logits(current, current_logits)[0]
        else:
            current_score = self._compute_energies(current)[0]
        scores = [current_score]
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

            if self.use_free_energy:
                effect_indices = torch.tensor(
                    [different_positions.index(position) for position in remaining],
                    device=current.device,
                )
                candidate_logits = (
                    current_logits[None, :, :]
                    + self._conditional_effects[effect_indices]
                )
                candidate_energies = self._compute_energies(candidate_sequences)
                candidate_scores = candidate_energies - self.slope * (
                    self._entropy_from_logits(candidate_logits)
                )
            else:
                candidate_scores = self._compute_energies(candidate_sequences)
            delta_scores = candidate_scores - current_score

            # Select the mutation producing the lowest-score intermediate.
            best = torch.argmin(delta_scores).item()
            position = remaining.pop(best)
            current = candidate_sequences[best]
            current_score = candidate_scores[best]
            if self.use_free_energy:
                current_logits = candidate_logits[best]

            order.append(position)
            scores.append(current_score)

        return order, torch.stack(scores)

    def _monte_carlo_path(
        self,
        start,
        end,
        different_positions,
        delta_score_function,
        score_function,
        initial_mutations=None,
    ):
        # Start a new chain from a random path. When make_path() is called on
        # an existing object without a seed, continue from its current order.
        if initial_mutations is None:
            order = self._rng.permutation(different_positions).tolist()
        else:
            order = [position - 1 for position in initial_mutations]
        path, scores = self._build_path(start, end, order)
        path_logits = None
        path_energies = None
        if self.use_free_energy:
            self._prepare_conditional_effects(start, end, different_positions)
            path_logits = self._ordered_conditional_logits(order)
            path_energies = self._compute_energies(path)

        if self.keep_history:
            current_score = score_function(scores).item()
            if initial_mutations is None or self.score_history is None:
                self.score_history = [current_score]

        if len(order) < 2:
            return order, scores

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
            # two exchanged residue positions. Calculate all their scores
            # changes together without rescoring the complete sequences.
            old_path = path[left + 1 : right + 1]
            delta_energy = self._delta_energy(
                old_path,
                new_path,
                torch.tensor(
                    [order[left], order[right]],
                    device=path.device,
                )[None, :].expand(len(new_path), -1),
            )
            if self.use_free_energy:
                mutation_to_effect = {
                    position: index
                    for index, position in enumerate(different_positions)
                }
                logits_change = (
                    self._conditional_effects[mutation_to_effect[order[right]]]
                    - self._conditional_effects[mutation_to_effect[order[left]]]
                )
                new_logits = path_logits[left + 1 : right + 1].clone()
                if len(new_logits) > 1:
                    new_logits[:-1] += logits_change
                delta_entropies = self._entropy_from_logits(new_logits)
                new_energies = (
                    path_energies[left + 1 : right + 1] + delta_energy
                )
                new_scores = new_energies - self.slope * delta_entropies
            else:
                new_scores = scores[left + 1 : right + 1] + delta_energy

            # The sampler is independent of the path objective. The selected
            # score function evaluates only the part changed by this proposal.
            delta_score = delta_score_function(
                scores,
                new_scores,
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
                # Keep the proposed order, sequences, and scores synchronized.
                order = new_order
                path[left + 1 : right + 1] = new_path
                scores[left + 1 : right + 1] = new_scores
                if self.use_free_energy:
                    path_energies[left + 1 : right + 1] = new_energies
                    if right - left > 1:
                        path_logits[left + 1 : right] += logits_change
                if self.keep_history:
                    current_score += delta_score

            if self.keep_history:
                self.score_history.append(current_score)

        return order, scores

    def _flat_path_score(self, scores):
        """Sum of squared score changes along the complete path."""
        return torch.sum(torch.diff(scores) ** 2)

    def _flat_delta_score(self, scores, new_scores, left, right):
        """Change in the squared score changes along the path."""
        # Include the unchanged sequence immediately before and after the
        # rebuilt segment because the jumps at both boundaries also change.
        old_window = scores[left : right + 2]
        new_window = torch.cat(
            (
                scores[left : left + 1],
                new_scores,
                scores[right + 1 : right + 2],
            )
        )
        return torch.sum(torch.diff(new_window) ** 2) - torch.sum(
            torch.diff(old_window) ** 2
        )

    def _mean_path_score(self, scores):
        """Mean selected score of every sequence along the path."""
        return torch.mean(scores)

    def _mean_delta_score(self, scores, new_scores, left, right):
        """Change in the mean selected score after a proposed swap."""
        # Endpoints and all sequences outside the rebuilt segment are
        # unchanged, so they cancel when comparing the two path means.
        old_scores = scores[left + 1 : right + 1]
        return torch.sum(new_scores - old_scores) / len(scores)

    def _build_path(self, start, end, order):
        # Apply the ordered mutations one at a time, retaining both endpoints.
        current = start.clone()
        path = [current.clone()]

        for position in order:
            current = current.clone()
            current[position] = end[position]
            path.append(current.clone())

        path = torch.stack(path)

        # Score the complete path in one batch. Conditional logits along a
        # mutation path are additive in its single-residue mutation effects.
        if self.use_free_energy:
            different_positions = torch.where(start != end)[0].tolist()
            self._prepare_conditional_effects(start, end, different_positions)
            logits = self._ordered_conditional_logits(order)
            scores = self._scores_from_logits(path, logits)
        else:
            scores = self._compute_energies(path)
        return path, scores

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

    def _prepare_conditional_effects(self, start, end, different_positions):
        """Precompute conditional-logit changes for each allowed mutation."""
        if self._conditional_effects is not None:
            return

        self._conditional_positions = different_positions.copy()
        self._reference_conditional_logits = compute_conditional_logits(
            start,
            self.params,
        )[0]
        coupling = self.params["coupling_matrix"]
        symmetric_coupling = 0.5 * (
            coupling + coupling.permute(2, 3, 0, 1)
        )

        effects = []
        for position in different_positions:
            old_aa = start[position]
            new_aa = end[position]
            effect = (
                symmetric_coupling[:, :, position, new_aa]
                - symmetric_coupling[:, :, position, old_aa]
            ).clone()
            effect[position] = 0
            effects.append(effect)

        if effects:
            self._conditional_effects = torch.stack(effects)
        else:
            self._conditional_effects = torch.empty(
                (0, start.shape[0], self.params["bias"].shape[1]),
                dtype=self.params["bias"].dtype,
                device=start.device,
            )

    def _ordered_conditional_logits(self, order):
        """Build all path logits from the reference plus mutation effects."""
        if not order:
            return self._reference_conditional_logits[None, :, :]
        position_to_effect = {
            position: index
            for index, position in enumerate(self._conditional_positions)
        }
        indices = torch.tensor(
            [position_to_effect[position] for position in order],
            dtype=torch.long,
            device=self._conditional_effects.device,
        )
        cumulative_effects = torch.cumsum(self._conditional_effects[indices], dim=0)
        return torch.cat(
            (
                self._reference_conditional_logits[None, :, :],
                self._reference_conditional_logits[None, :, :] + cumulative_effects,
            ),
            dim=0,
        )

    def _entropy_from_logits(self, logits):
        log_probabilities = torch.log_softmax(logits, dim=-1)
        probabilities = torch.exp(log_probabilities)
        return -(probabilities * log_probabilities).sum(dim=-1).sum(dim=-1)

    def _scores_from_logits(self, sequences, logits):
        energies = self._compute_energies(sequences)
        return energies - self.slope * self._entropy_from_logits(logits)

    def _compute_entropies(self, sequences):
        """Sum H(S_i | S_-i) from the Potts conditional distributions."""
        return compute_conditional_entropies(sequences, self.params)

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
    and ``mean`` paths, DCA terms are evaluated once before sampling.
    Monte Carlo then operates only on the mutation order and score increments.
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
        use_free_energy=False,
        slope=1,
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
            use_free_energy=use_free_energy,
            slope=slope,
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
        """Run random or greedy paths normally, or use the fast sampler."""
        if self.algorithm == "random":
            return super()._run_algorithm(
                start,
                end,
                different_positions,
                initial_mutations,
            )
        if self.use_free_energy:
            # Conditional entropy depends on every residue in each sequence,
            # so the energy-only precomputed effects do not apply.
            return super()._run_algorithm(
                start,
                end,
                different_positions,
                initial_mutations,
            )
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
        if self.algorithm == "mean":
            return self._fast_monte_carlo_path(
                start,
                end,
                different_positions,
                self._mean_step_delta_score,
                self._mean_step_score,
                initial_mutations,
            )

        raise ValueError("algorithm must be 'random', 'greedy', 'flat', or 'mean'.")

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
        score_function,
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

        score_steps = self._ordered_score_steps(order)

        if self.keep_history:
            current_score = score_function(score_steps).item()
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

                # Swapping two mutation times changes only the score steps in
                # this interval. Update them directly from the pair effects.
                new_score_steps = score_steps[left : right + 1].clone()
                new_score_steps[0] = score_steps[right] - self.pair_effects[
                    mutation_right,
                    mutation_left,
                ]
                new_score_steps[-1] = score_steps[left] + self.pair_effects[
                    mutation_left,
                    mutation_right,
                ]

                if len(middle) > 0:
                    new_score_steps[0] -= self.pair_effects[
                        mutation_right,
                        middle,
                    ].sum()
                    new_score_steps[-1] += self.pair_effects[
                        mutation_left,
                        middle,
                    ].sum()
                    new_score_steps[1:-1] += (
                        self.pair_effects[middle, mutation_right]
                        - self.pair_effects[middle, mutation_left]
                    )

                delta_score = delta_score_function(
                    score_steps,
                    new_score_steps,
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
                    score_steps[left : right + 1] = new_score_steps
                    if self.keep_history:
                        current_score += delta_score

                if self.keep_history:
                    self.score_history.append(current_score)

        scores = torch.cat(
            (
                self.reference_energy[None],
                self.reference_energy + torch.cumsum(score_steps, dim=0),
            )
        )
        return [different_positions[index] for index in order], scores

    def _ordered_score_steps(self, order):
        """Calculate the score change at each mutation time."""
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

    def _flat_step_score(self, score_steps):
        """Flat-path objective from score increments."""
        return torch.sum(score_steps**2)

    def _flat_step_delta_score(
        self,
        score_steps,
        new_score_steps,
        left,
        right,
    ):
        """Change in the flat-path objective after a proposed swap."""
        return torch.sum(new_score_steps**2) - torch.sum(
            score_steps[left : right + 1] ** 2
        )

    def _mean_step_score(self, score_steps):
        """Mean selected score of every sequence along the path."""
        return torch.mean(
            torch.cat(
                (
                    self.reference_energy[None],
                    self.reference_energy + torch.cumsum(score_steps, dim=0),
                )
            )
        )

    def _mean_step_delta_score(
        self,
        score_steps,
        new_score_steps,
        left,
        right,
    ):
        """Change in mean path score after a proposed swap."""
        changed_steps = (
            new_score_steps - score_steps[left : right + 1]
        )
        return torch.sum(torch.cumsum(changed_steps, dim=0)) / (
            len(score_steps) + 1
        )
