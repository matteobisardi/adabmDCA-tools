import tempfile
import unittest
from pathlib import Path

import torch

from utils_dca import MultipleSequenceAlignment, make_setup
from utils_dca.fasta import import_from_fasta_keep_order


class CoreTests(unittest.TestCase):
    def test_default_setup(self):
        setup = make_setup(device="cpu")
        self.assertEqual(setup["tokens"], "-ACDEFGHIKLMNPQRSTVWY")
        self.assertEqual(setup["q"], 21)

    def test_msa_components_and_cache_invalidation(self):
        encoded = torch.tensor([[1, 2, 3], [1, 2, 4], [1, 3, 4]])
        onehot = torch.nn.functional.one_hot(encoded, num_classes=21).float()
        msa = MultipleSequenceAlignment.from_onehot(onehot)
        self.assertEqual(msa.compute_pca(n_components=2).shape, (3, 2))

        msa.remove_items([0], axis="columns")
        self.assertIsNone(msa.weights)
        self.assertIsNone(msa.Meff)
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


if __name__ == "__main__":
    unittest.main()
