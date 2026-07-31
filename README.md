# utils_dca

Local utilities built on top of `adabmDCA` for handling multiple-sequence alignments (MSAs), protein sequences, and related helper functions.
The package bundles two main modules:

- `classes.py`: high-level wrappers such as `MultipleSequenceAlignment` and `ProteinSequence`, plus helpers for Deep Mutational Scanning workflows.
- `funcs.py`: standalone helpers for FASTA import, gap statistics, and one-hot manipulations.

## Install (editable, local only)

pip install -e .

Make sure `adabmDCA` and its dependencies are already available in the same environment (either via `pip install adabmDCA` or by adding your local copy to `PYTHONPATH`).

## Quick Start

from utils_dca import MultipleSequenceAlignment, import_unaligned_fasta

setup = {
    "device": "cpu",
    "dtype": "float32",
    "tokens": "-ACDEFGHIKLMNPQRSTVWY",
    "q": 21,  # number of tokens including gap
}

msa = MultipleSequenceAlignment.from_path("example_msa.fasta", setup)
msa.compute_gap_frequency()
msa.summary()

headers, sequences = import_unaligned_fasta("example_unaligned.fasta",
                                            tokens=setup["tokens"],
                                            filter_sequences=True)

### Key Features

- MultipleSequenceAlignment
  - Load aligned FASTA files, numerically encode sequences, and compute one-hot tensors.
  - Filter out columns or rows by index or gap-frequency threshold.
  - Compute sequence weights (`compute_weights_cls`) and gap statistics.

- ProteinSequence
  - Encode a single aligned sequence, access aligned/unaligned forms, and map aligned ↔ unaligned positions.
  - Access a `DeepMutationalScanning` helper via `protein.dms`.

- Utility Functions (from funcs.py)
  - `compute_gap_frequency(msa_oh)`: gap fraction per column for a one-hot MSA.
  - `compute_seqID(a1, single_seq)`: sequence identity scores against a reference.
  - `count_gaps(msa)`: number of gaps per sequence (works for single or batched inputs).
  - `import_unaligned_fasta(...)`: read FASTA files with optional token filtering and duplicate removal.

## Testing

A placeholder `test.py` exists; replace it with real tests as you formalize the package (e.g., smoke tests for `MultipleSequenceAlignment.from_path`).

## License

Proprietary / internal use only. Update this section if you plan to distribute the package.
