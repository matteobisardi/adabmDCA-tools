import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from adabmDCA_tools import SequencePath, SequencePathFast, make_setup


class SequencePathFastTests(unittest.TestCase):
    def setUp(self):
        self.setup = make_setup(alphabet="-ABC", device="cpu")
        self.wildtype1 = "ABC-AB"
        self.wildtype2 = "BC-AAB"

        generator = torch.Generator().manual_seed(23)
        self.params = {
            "bias": torch.randn(6, 4, generator=generator),
            # Non-symmetric couplings and their diagonal terms are retained.
            "coupling_matrix": torch.randn(
                6,
                4,
                6,
                4,
                generator=generator,
            ),
        }

    def test_precomputed_effects_reconstruct_every_hybrid(self):
        path = SequencePathFast.flat(
            self.wildtype1,
            self.wildtype2,
            self.params,
            steps=0,
            seed=3,
            setup=self.setup,
        )
        start, end = path._prepare_wildtypes()
        positions = torch.where(start != end)[0]

        for bits in itertools.product((0, 1), repeat=len(positions)):
            sequence = start.clone()
            active = [index for index, bit in enumerate(bits) if bit]
            if len(active) > 0:
                sequence[positions[active]] = end[positions[active]]

            reconstructed = path.reference_energy.clone()
            if len(active) > 0:
                reconstructed += path.single_effects[active].sum()
                reconstructed += 0.5 * path.pair_effects[active][:, active].sum()

            np.testing.assert_allclose(
                reconstructed.numpy(),
                path._compute_energies(sequence)[0].cpu().numpy(),
                atol=1e-5,
            )

    def test_fast_and_standard_samplers_match(self):
        for algorithm in ("flat", "mean_energy"):
            standard = SequencePath(
                self.wildtype1,
                self.wildtype2,
                self.params,
                algorithm=algorithm,
                beta=0.7,
                steps=500,
                seed=11,
                setup=self.setup,
                keep_history=True,
            )
            fast = SequencePathFast(
                self.wildtype1,
                self.wildtype2,
                self.params,
                algorithm=algorithm,
                beta=0.7,
                steps=500,
                seed=11,
                setup=self.setup,
                keep_history=True,
            )

            self.assertEqual(fast.mutations, standard.mutations)
            np.testing.assert_allclose(fast.energies, standard.energies, atol=1e-5)
            np.testing.assert_allclose(
                fast.score_history,
                standard.score_history,
                atol=1e-4,
            )

    def test_fast_free_energy_paths_match_standard_paths(self):
        for algorithm in ("flat", "mean"):
            standard = SequencePath(
                self.wildtype1,
                self.wildtype2,
                self.params,
                algorithm=algorithm,
                beta=0.7,
                steps=40,
                seed=11,
                setup=self.setup,
                keep_history=True,
                use_free_energy=True,
                slope=0.8,
            )
            fast = SequencePathFast(
                self.wildtype1,
                self.wildtype2,
                self.params,
                algorithm=algorithm,
                beta=0.7,
                steps=40,
                seed=11,
                setup=self.setup,
                keep_history=True,
                use_free_energy=True,
                slope=0.8,
            )

            self.assertEqual(fast.mutations, standard.mutations)
            np.testing.assert_allclose(fast.energies, standard.energies, atol=1e-5)
            np.testing.assert_allclose(fast.entropies, standard.entropies, atol=1e-5)
            np.testing.assert_allclose(
                fast.free_energies, standard.free_energies, atol=1e-5
            )
            np.testing.assert_allclose(
                fast.score_history, standard.score_history, atol=1e-4
            )

    def test_fast_path_inherits_mutation_distances(self):
        path = SequencePathFast.random(
            self.wildtype1,
            self.wildtype2,
            self.params,
            seed=1,
            setup=self.setup,
        )
        other = SequencePath.random(
            self.wildtype1,
            self.wildtype2,
            self.params,
            seed=2,
            setup=self.setup,
        )

        self.assertGreaterEqual(path.distance_k(other), 0.0)
        self.assertLessEqual(path.distance_k(other), 1.0)
        self.assertEqual(len(path.distance_introduction(other)), 4)

    def test_history_continuation_and_restart(self):
        path = SequencePathFast.mean_energy(
            self.wildtype1,
            self.wildtype2,
            self.params,
            steps=20,
            seed=7,
            setup=self.setup,
            keep_history=True,
        )
        first_history = path.score_history.copy()

        self.assertEqual(len(first_history), 21)
        self.assertAlmostEqual(
            first_history[-1],
            np.mean(path.energies),
            places=5,
        )

        path.make_path()
        self.assertEqual(len(path.score_history), 41)

        path.make_path(seed=7)
        self.assertEqual(len(path.score_history), 21)
        np.testing.assert_allclose(path.score_history, first_history, atol=1e-5)

    def test_from_file_returns_fast_path(self):
        with tempfile.TemporaryDirectory() as directory:
            fasta = Path(directory) / "path.fasta"
            fasta.write_text(
                ">start\nABC-AB\n"
                ">one\nBBC-AB\n"
                ">two\nBCC-AB\n"
                ">three\nBC--AB\n"
                ">four\nBC-AAB\n"
            )
            path = SequencePathFast.from_file(
                fasta,
                self.params,
                setup=self.setup,
            )

        self.assertIsInstance(path, SequencePathFast)
        self.assertEqual(path.mutations, [1, 2, 3, 4])
        self.assertIsNone(path.single_effects)
        self.assertIsNone(path.pair_effects)


if __name__ == "__main__":
    unittest.main()
