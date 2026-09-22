import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import torch

from adabmDCA.statmech import compute_energy
from adabmDCA_tools import (
    MultipleSequenceAlignment,
    ProteinSequence,
    SequencePath,
    make_setup,
)


class SequencePathTests(unittest.TestCase):
    def setUp(self):
        self.setup = make_setup(alphabet="-AC", device="cpu")
        self.wildtype1 = "AAA"
        self.wildtype2 = "CCC"

    def _model(self):
        bias = torch.zeros(3, 3)
        # Since energy is minus the model score, the greedy path should take
        # positions 1, 2, and 3 in decreasing bias order.
        bias[:, 2] = torch.tensor([3.0, 2.0, 1.0])
        return {
            "bias": bias,
            "coupling_matrix": torch.zeros(3, 3, 3, 3),
        }

    def _path_sequences(self, path):
        msa = path.to_msa()
        return [
            "".join(self.setup["tokens"][aa] for aa in sequence)
            for sequence in msa.seqs
        ]

    def test_greedy_path_and_msa_conversion(self):
        path = SequencePath.greedy(
            self.wildtype1,
            self.wildtype2,
            self._model(),
            setup=self.setup,
        )

        self.assertEqual(path.mutations, [1, 2, 3])
        self.assertEqual(self._path_sequences(path), ["AAA", "CAA", "CCA", "CCC"])
        self.assertEqual(len(path.mutations), 3)
        self.assertEqual(len(path), 4)
        np.testing.assert_allclose(path.energies, [0.0, -3.0, -5.0, -6.0])

        msa = path.to_msa()
        self.assertIsInstance(msa, MultipleSequenceAlignment)
        self.assertEqual(
            msa.headers.tolist(),
            ["wildtype1", "step_1_A1C", "step_2_A2C", "wildtype2"],
        )
        self.assertEqual(
            msa.seqs.tolist(),
            [[1, 1, 1], [2, 1, 1], [2, 2, 1], [2, 2, 2]],
        )
        self.assertIs(path.to_msa(), path.msa)
        self.assertEqual(path[0][0], "wildtype1")
        self.assertEqual(path[0][1].tolist(), [1, 1, 1])
        self.assertEqual(path[3][0], "wildtype2")
        self.assertEqual(path[3][1].tolist(), [2, 2, 2])

        old_msa = path.msa
        message = StringIO()
        with redirect_stdout(message):
            self.assertIs(path.make_path(), path)
        self.assertIs(path.msa, old_msa)
        self.assertIn("greedy path is deterministic", message.getvalue())

    def test_write_path_to_file(self):
        path = SequencePath.greedy(
            self.wildtype1,
            self.wildtype2,
            self._model(),
            setup=self.setup,
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "path.fasta"
            path.write_to_file(output)
            content = output.read_text()

        self.assertIn(">wildtype1\nAAA", content)
        self.assertIn(">step_1_A1C\nCAA", content)
        self.assertIn(">wildtype2\nCCC", content)

    def test_import_path_from_fasta(self):
        with tempfile.TemporaryDirectory() as directory:
            fasta = Path(directory) / "path.fasta"
            fasta.write_text(
                ">start\nAAA\n"
                ">first\nCAA\n"
                ">second\nCCA\n"
                ">end\nCCC\n"
            )
            path = SequencePath.from_file(
                fasta,
                self._model(),
                setup=self.setup,
            )

            self.assertEqual(path.wildtype1, "AAA")
            self.assertEqual(path.wildtype2, "CCC")
            self.assertEqual(path.mutations, [1, 2, 3])
            self.assertEqual(path.msa.headers.tolist(), ["start", "first", "second", "end"])
            np.testing.assert_allclose(path.energies, [0.0, -3.0, -5.0, -6.0])

            fasta.write_text(">start\nAAA\n>jump\nCCA\n>end\nCCC\n")
            with self.assertRaisesRegex(ValueError, "exactly one position"):
                SequencePath.from_file(
                    fasta,
                    self._model(),
                    setup=self.setup,
                )

            fasta.write_text(">start\nAAA\n>wrong\n-AA\n>end\nCAA\n")
            with self.assertRaisesRegex(ValueError, "directly toward wildtype2"):
                SequencePath.from_file(
                    fasta,
                    self._model(),
                    setup=self.setup,
                )

    def test_supported_wildtype_inputs(self):
        protein = ProteinSequence("AAA", setup=self.setup)
        encoded = torch.tensor([2, 2, 2])

        path = SequencePath.greedy(
            protein,
            encoded,
            self._model(),
            setup=self.setup,
        )
        self.assertEqual(path.wildtype1, "AAA")
        self.assertEqual(path.wildtype2, "CCC")

        with tempfile.TemporaryDirectory() as directory:
            fasta = Path(directory) / "wildtype.fasta"
            fasta.write_text(">VIM-2\nAAA\n")
            path = SequencePath.greedy(
                str(fasta),
                [2, 2, 2],
                self._model(),
                setup=self.setup,
            )

        self.assertEqual(path.wildtype1, "AAA")
        self.assertEqual(path.wildtype2, "CCC")

    def test_monte_carlo_paths_are_valid_and_reproducible(self):
        model = self._model()
        flat1 = SequencePath.flat(
            self.wildtype1,
            self.wildtype2,
            model,
            beta=2.0,
            steps=100,
            seed=9,
            setup=self.setup,
        )
        flat2 = SequencePath.flat(
            self.wildtype1,
            self.wildtype2,
            model,
            beta=2.0,
            steps=100,
            seed=9,
            setup=self.setup,
        )
        mean_path = SequencePath.mean_energy(
            self.wildtype1,
            self.wildtype2,
            model,
            beta=2.0,
            steps=100,
            seed=4,
            setup=self.setup,
        )

        self.assertEqual(flat1.mutations, flat2.mutations)
        np.testing.assert_allclose(flat1.energies, flat2.energies)
        first_mutations = flat1.mutations.copy()
        first_energies = flat1.energies.copy()
        first_msa = flat1.msa

        self.assertIs(flat1.make_path(), flat1)
        self.assertIsNot(flat1.msa, first_msa)

        flat1.make_path(seed=9)
        self.assertEqual(flat1.mutations, first_mutations)
        np.testing.assert_allclose(flat1.energies, first_energies)
        self.assertEqual(sorted(flat1.mutations), [1, 2, 3])
        self.assertEqual(sorted(mean_path.mutations), [1, 2, 3])
        flat_sequences = self._path_sequences(flat1)
        self.assertEqual(flat_sequences[0], "AAA")
        self.assertEqual(flat_sequences[-1], "CCC")
        self.assertTrue(np.isfinite(flat1.energies).all())
        self.assertTrue(np.isfinite(mean_path.energies).all())

    def test_make_path_continues_or_restarts_monte_carlo(self):
        model = self._model()
        path = SequencePath.flat(
            self.wildtype1,
            self.wildtype2,
            model,
            steps=0,
            seed=9,
            setup=self.setup,
        )
        initial_mutations = path.mutations.copy()

        # Without a seed, sampling starts from the current mutation order.
        path.make_path()
        self.assertEqual(path.mutations, initial_mutations)

        # With a seed, sampling restarts from the corresponding random order.
        reference = SequencePath.flat(
            self.wildtype1,
            self.wildtype2,
            model,
            steps=0,
            seed=4,
            setup=self.setup,
        )
        path.make_path(seed=4)
        self.assertEqual(path.mutations, reference.mutations)

        random_seed_path = SequencePath.flat(
            self.wildtype1,
            self.wildtype2,
            model,
            steps=0,
            setup=self.setup,
        )
        self.assertIsInstance(random_seed_path.seed, int)

    def test_monte_carlo_score_history(self):
        flat = SequencePath.flat(
            self.wildtype1,
            self.wildtype2,
            self._model(),
            steps=20,
            seed=7,
            setup=self.setup,
            keep_history=True,
        )
        first_history = flat.score_history.copy()

        self.assertEqual(len(flat.score_history), 21)
        self.assertAlmostEqual(
            flat.score_history[-1],
            np.sum(np.diff(flat.energies) ** 2),
            places=5,
        )

        # Continuing adds one score per additional Monte Carlo step.
        flat.make_path()
        self.assertEqual(len(flat.score_history), 41)

        # Restarting resets the history and reproduces the seeded run.
        flat.make_path(seed=7)
        self.assertEqual(len(flat.score_history), 21)
        np.testing.assert_allclose(flat.score_history, first_history)

        mean = SequencePath.mean_energy(
            self.wildtype1,
            self.wildtype2,
            self._model(),
            steps=20,
            seed=7,
            setup=self.setup,
            keep_history=True,
        )
        self.assertEqual(len(mean.score_history), 21)
        self.assertAlmostEqual(
            mean.score_history[-1],
            np.mean(mean.energies),
            places=5,
        )

        without_history = SequencePath.flat(
            self.wildtype1,
            self.wildtype2,
            self._model(),
            steps=1,
            seed=7,
            setup=self.setup,
        )
        self.assertIsNone(without_history.score_history)

    def test_path_energies_match_adabmdca(self):
        generator = torch.Generator().manual_seed(12)
        model = {
            "bias": torch.randn(3, 3, generator=generator),
            # Deliberately non-symmetric couplings ensure that the complete
            # adabmDCA energy convention is used.
            "coupling_matrix": torch.randn(3, 3, 3, 3, generator=generator),
        }
        sampled = SequencePath.flat(
            self.wildtype1,
            self.wildtype2,
            model,
            beta=0.5,
            steps=100,
            seed=21,
            setup=self.setup,
        )
        sampled_reference = compute_energy(sampled.to_msa().onehot(), model)
        self.assertTrue(
            torch.allclose(
                torch.as_tensor(sampled.energies),
                sampled_reference,
                atol=1e-5,
            )
        )

        old_sequences = torch.tensor([[1, 1, 1], [1, 2, 1]])
        new_sequences = torch.tensor([[2, 1, 1], [1, 2, 2]])
        changed_positions = torch.tensor([[0], [2]])
        delta = sampled._delta_energy(
            old_sequences,
            new_sequences,
            changed_positions,
        )
        reference_delta = (
            sampled._compute_energies(new_sequences)
            - sampled._compute_energies(old_sequences)
        )
        self.assertTrue(torch.allclose(delta, reference_delta, atol=1e-5))

        old_sequences = torch.tensor([[1, 1, 1], [1, 2, 1]])
        new_sequences = torch.tensor([[2, 1, 2], [2, 2, 2]])
        changed_positions = torch.tensor([[0, 2], [0, 2]])
        delta = sampled._delta_energy(
            old_sequences,
            new_sequences,
            changed_positions,
        )
        reference_delta = (
            sampled._compute_energies(new_sequences)
            - sampled._compute_energies(old_sequences)
        )
        self.assertTrue(torch.allclose(delta, reference_delta, atol=1e-5))

        # The stored sequence at each step must be exactly the cumulative
        # application of the public 1-based mutation order.
        rebuilt = list(self.wildtype1)
        rebuilt_path = ["".join(rebuilt)]
        target = self.wildtype2
        for position in sampled.mutations:
            rebuilt[position - 1] = target[position - 1]
            rebuilt_path.append("".join(rebuilt))
        self.assertEqual(self._path_sequences(sampled), rebuilt_path)

    def test_constructor_validates_endpoints_and_model(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            SequencePath("AA", "CCC", self._model(), setup=self.setup)

        wrong_model = {
            "bias": torch.zeros(2, 3),
            "coupling_matrix": torch.zeros(2, 3, 2, 3),
        }
        with self.assertRaisesRegex(ValueError, "DCA model"):
            SequencePath(
                self.wildtype1,
                self.wildtype2,
                wrong_model,
                setup=self.setup,
            )


if __name__ == "__main__":
    unittest.main()
