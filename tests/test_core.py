import tempfile
import unittest
from pathlib import Path

import torch

from adabmDCA_tools import (
    MultipleSequenceAlignment,
    ProteinSequence,
    compute_gap_frequency,
    import_unaligned_fasta,
    make_setup,
)
from adabmDCA_tools.fasta import import_from_fasta_keep_order


class CoreTests(unittest.TestCase):
    def test_default_setup(self):
        setup = make_setup(device="cpu")
        self.assertEqual(setup["tokens"], "-ACDEFGHIKLMNPQRSTVWY")
        self.assertEqual(setup["q"], 21)

    def test_msa_components_and_cache_invalidation(self):
        encoded = torch.tensor([[1, 2, 3], [1, 2, 4], [1, 3, 4]])
        onehot = torch.nn.functional.one_hot(encoded, num_classes=21).float()
        msa = MultipleSequenceAlignment.from_onehot(
            onehot, setup=make_setup(device="cpu")
        )
        self.assertEqual(msa.enc.device.type, "cpu")
        self.assertIsNone(msa._onehot_cache)

        temporary_onehot = msa.onehot()
        self.assertEqual(temporary_onehot.device.type, "cpu")
        self.assertIsNone(msa._onehot_cache)
        cached_onehot = msa.onehot(cache=True)
        self.assertIs(msa._onehot_cache, cached_onehot)
        self.assertIs(msa.get_onehot(cache=True), cached_onehot)
        msa.clear_cache(empty_mps_cache=False)
        self.assertIsNone(msa._onehot_cache)

        self.assertEqual(msa.compute_pca(n_components=2).shape, (3, 2))
        self.assertIsNone(msa._onehot_cache)

        msa.remove_items([0], axis="columns")
        self.assertIsNotNone(msa.weights)
        self.assertEqual(msa.Meff, 3.0)
        self.assertTrue(torch.equal(msa.gap_freq, torch.zeros(2)))
        self.assertEqual(msa.compute_pca(n_components=2).shape, (3, 2))

    def test_rna_fasta_round_trip(self):
        setup = make_setup(alphabet="rna", device="cpu")
        encoded = torch.tensor([[1, 2, 3, 4]])
        onehot = torch.nn.functional.one_hot(encoded, num_classes=5).float()
        msa = MultipleSequenceAlignment.from_onehot(onehot, setup=setup)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rna.fasta"
            msa.write_to_file(path)
            _, sequences = import_from_fasta_keep_order(path)

        self.assertEqual(sequences.tolist(), ["ACGU"])

    def test_import_unaligned_fasta_preserves_duplicates_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequences.fasta"
            path.write_text(">first\nACDE\n>duplicate\nACDE\n>other\nACDF\n")

            headers, sequences = import_unaligned_fasta(path)
            dedup_headers, dedup_sequences = import_unaligned_fasta(
                path, remove_duplicates=True
            )

        self.assertEqual(headers, ["first", "duplicate", "other"])
        self.assertEqual(sequences, ["ACDE", "ACDE", "ACDF"])
        self.assertEqual(dedup_headers, ["first", "other"])
        self.assertEqual(dedup_sequences, ["ACDE", "ACDF"])

    def test_protein_sequence_from_path_preserves_alignment(self):
        setup = make_setup(device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protein.fasta"
            path.write_text(">VIM-2 aligned\nACd\n-E\n")

            protein = ProteinSequence.from_path(path, setup=setup)

        self.assertEqual(protein.name, "VIM-2 aligned")
        self.assertEqual(protein.afa, "ACd-E")
        self.assertEqual(protein.aligned.get_string(), "AC-E")
        self.assertEqual(protein.unaligned.get_string(), "ACDE")

    def test_protein_sequence_from_path_requires_one_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proteins.fasta"
            path.write_text(">first\nACDE\n>second\nACDF\n")

            with self.assertRaisesRegex(ValueError, "exactly one FASTA sequence"):
                ProteinSequence.from_path(path)

    def test_remove_sequences_close_to_wildtype(self):
        setup = make_setup(device="cpu")
        msa = MultipleSequenceAlignment(
            ["wt", "one_mutation", "distant"],
            [[1, 2, 3, 4], [1, 2, 3, 5], [5, 6, 7, 8]],
            setup=setup,
        )
        wildtype = MultipleSequenceAlignment(["wildtype"], [[1, 2, 3, 4]], setup=setup)

        removed = msa.remove_sequences_close_to_wildtype(wildtype, 0.25)

        self.assertEqual(removed.tolist(), [0, 1])
        self.assertEqual(msa.headers.tolist(), ["distant"])
        self.assertEqual(msa.seqs.tolist(), [[5, 6, 7, 8]])

    def test_estimate_potts_snr(self):
        setup = make_setup(alphabet="-AB", device="cpu")
        msa = MultipleSequenceAlignment(
            ["s1", "s2", "s3", "s4"],
            [[0, 0], [1, 0], [2, 0], [2, 0]],
            setup=setup,
        )
        msa.weights = torch.ones(4)
        msa.Meff = 4.0
        model = {
            "bias": torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]),
            "coupling_matrix": torch.zeros(2, 3, 2, 3),
        }

        snr = msa.estimate_potts_snr(model)

        # Energies are [0, -1, -2, -2]: chi^2 = 0.6875 and theta = 8.
        self.assertAlmostEqual(snr, 4 * 0.6875 / 8)

    def test_estimate_potts_snr_validates_model_dimensions(self):
        msa = MultipleSequenceAlignment(["sequence"], [[1, 2]])
        model = {
            "bias": torch.zeros(3, 21),
            "coupling_matrix": torch.zeros(3, 21, 3, 21),
        }

        with self.assertRaisesRegex(ValueError, "Model dimensions do not match"):
            msa.estimate_potts_snr(model)

    def test_remove_sequences_close_to_aligned_protein(self):
        setup = make_setup(device="cpu")
        protein = ProteinSequence("ACd-E", setup=setup)
        msa = MultipleSequenceAlignment(
            ["exact", "one_mutation", "distant"],
            [protein.num.tolist(), [1, 2, 0, 5], [5, 6, 7, 8]],
            setup=setup,
        )

        removed = msa.remove_sequences_close_to_wildtype(protein, 0.0)

        # The lowercase deleted residue is absent from the aligned sequence.
        self.assertEqual(protein.aligned.get_string(), "AC-E")
        self.assertEqual(removed.tolist(), [0])
        self.assertEqual(msa.headers.tolist(), ["one_mutation", "distant"])

    def test_gap_frequency_uses_sequence_weights(self):
        msa = MultipleSequenceAlignment(
            ["duplicate_1", "duplicate_2", "other"],
            [[0, 1], [0, 1], [1, 1]],
            setup=make_setup(device="cpu"),
        )

        gap_freq = msa.compute_gap_frequency(th=0.8)

        # The duplicate sequences each have weight 1/2, while the other
        # sequence has weight 1.  The first column is therefore weighted as
        # (1/2 + 1/2) / (1/2 + 1/2 + 1) = 1/2 rather than 2/3.
        self.assertTrue(torch.allclose(gap_freq, torch.tensor([0.5, 0.0])))
        self.assertTrue(torch.allclose(msa.weights, torch.tensor([0.5, 0.5, 1.0])))
        self.assertEqual(msa.Meff, 2.0)

        onehot_gap_freq = compute_gap_frequency(msa.onehot(), th=0.8)
        self.assertTrue(torch.allclose(onehot_gap_freq, gap_freq))

    def test_remove_sequences_rejects_protein_with_wrong_alignment_length(self):
        msa = MultipleSequenceAlignment(["sequence"], [[1, 2, 3, 4]])
        protein = ProteinSequence("ACD")

        with self.assertRaisesRegex(ValueError, "Alignment length mismatch"):
            msa.remove_sequences_close_to_wildtype(protein, 0.1)

    def test_concatenate_msa_fast_joins_valid_caches(self):
        setup = make_setup(device="cpu")
        left = MultipleSequenceAlignment(["a", "b"], [[1, 0], [1, 2]], setup=setup)
        right = MultipleSequenceAlignment(["c"], [[0, 2]], setup=setup)
        left.compute_gap_frequency(th=0.8)
        right.compute_gap_frequency(th=0.8)
        left.onehot(cache=True)
        right.onehot(cache=True)
        left.weights = torch.ones(2)
        left.Meff = 2.0
        left._V = torch.ones(42, 1)
        left._pca_mean = torch.ones(1, 42)

        result = left.concatenate(right, fast=True)

        self.assertIs(result, left)
        self.assertEqual(left.headers.tolist(), ["a", "b", "c"])
        self.assertEqual(left.seqs.tolist(), [[1, 0], [1, 2], [0, 2]])
        self.assertIsNone(left.gap_freq)
        self.assertEqual(left._onehot_cache.shape, (3, 2, 21))
        self.assertIsNone(left.weights)
        self.assertIsNone(left.Meff)
        self.assertIsNone(left._V)
        self.assertIsNone(left._pca_mean)

    def test_concatenate_msa_fast_does_not_create_missing_caches(self):
        setup = make_setup(device="cpu")
        left = MultipleSequenceAlignment(["a"], [[1, 2]], setup=setup)
        right = MultipleSequenceAlignment(["b"], [[2, 1]], setup=setup)
        left.compute_gap_frequency(th=0.8)

        left.concatenate(right, fast=True)

        self.assertIsNone(left.gap_freq)
        self.assertIsNone(left._onehot_cache)
        self.assertIsNone(left.weights)

    def test_concatenate_msa_validates_compatibility(self):
        protein = MultipleSequenceAlignment(["a"], [[1, 2]])
        different_length = MultipleSequenceAlignment(["b"], [[1, 2, 3]])
        rna = MultipleSequenceAlignment(["c"], [[1, 2]], setup=make_setup(alphabet="rna"))

        with self.assertRaises(ValueError):
            protein.concatenate(different_length)
        with self.assertRaises(ValueError):
            protein.concatenate(rna)

    def test_concatenate_many_does_not_modify_inputs(self):
        setup = make_setup(device="cpu")
        first = MultipleSequenceAlignment(["a"], [[1, 2]], setup=setup)
        second = MultipleSequenceAlignment(["b"], [[2, 1]], setup=setup)
        first.compute_gap_frequency(th=0.8)
        second.compute_gap_frequency(th=0.8)

        combined = MultipleSequenceAlignment.concatenate_many(first, second)

        self.assertEqual(combined.headers.tolist(), ["a", "b"])
        self.assertEqual(combined.seqs.tolist(), [[1, 2], [2, 1]])
        self.assertEqual(first.headers.tolist(), ["a"])
        self.assertEqual(first.seqs.tolist(), [[1, 2]])
        self.assertEqual(second.headers.tolist(), ["b"])
        self.assertEqual(second.seqs.tolist(), [[2, 1]])
        self.assertIsNot(combined.enc, first.enc)
        self.assertIsNotNone(combined.weights)
        self.assertIsNotNone(combined.gap_freq)

    def test_concatenate_many_single_alignment_returns_copy(self):
        original = MultipleSequenceAlignment(["a"], [[1, 2]])

        combined = MultipleSequenceAlignment.concatenate_many(original)
        combined.headers[0] = "changed"
        combined.enc[0, 0] = 3

        self.assertEqual(original.headers.tolist(), ["a"])
        self.assertEqual(original.seqs.tolist(), [[1, 2]])

    def test_protein_sequence_views_update_parent(self):
        protein = ProteinSequence("ACd-E")

        protein.aligned[2] = "V"
        self.assertEqual(protein.afa, "AVd-E")
        self.assertEqual(protein.unaligned.get_string(), "AVDE")
        self.assertEqual(protein.enc.shape, (4,))

        del protein.aligned[3]
        self.assertEqual(protein.afa, "AVE")
        self.assertEqual(protein.L, 3)

        protein.unaligned.insert(2, "G")
        self.assertEqual(protein.afa, "AGVE")
        self.assertEqual(protein.aligned.get_string(), "AGVE")
        self.assertEqual(protein.enc.shape, (4,))


if __name__ == "__main__":
    unittest.main()
