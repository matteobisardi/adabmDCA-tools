# utils_dca

Local utilities built on top of `adabmDCA` for handling multiple-sequence alignments (MSAs), protein sequences, and related helper functions.
The package is organized into focused modules:

- `msa.py`: `MultipleSequenceAlignment`.
- `protein.py`: `ProteinSequence`.
- `dms.py`: deep-mutational-scanning tools, loaded only when `protein.dms` is accessed.
- `fasta.py`: FASTA readers.
- `metrics.py`: standalone numerical helpers.

The old `classes.py` and `funcs.py` import paths remain available for existing code.

## Install (editable, local only)

```bash
python -m pip install -e .
```

Make sure `adabmDCA` and its dependencies are already available in the same environment (either via `pip install adabmDCA` or by adding your local copy to `PYTHONPATH`).

## Quick Start

```python
from utils_dca import MultipleSequenceAlignment, import_unaligned_fasta

msa = MultipleSequenceAlignment.from_path("example_msa.fasta")
msa.compute_gap_frequency()
msa.summary()

# For explicit control over the alphabet, device, or dtype:
from utils_dca import make_setup
setup = make_setup(alphabet="protein", device="cpu", dtype="float32")
msa = MultipleSequenceAlignment.from_path("example_msa.fasta", setup)

headers, sequences = import_unaligned_fasta("example_unaligned.fasta",
                                            tokens=setup["tokens"],
                                            filter_sequences=True)
```

### Key Features

- MultipleSequenceAlignment
  - Load aligned FASTA files, numerically encode sequences, and compute one-hot tensors.
  - Filter out columns or rows by index or gap-frequency threshold.
  - Compute sequence weights (`compute_weights_cls`) and gap statistics.

- ProteinSequence
  - Encode a single aligned sequence, access aligned/unaligned forms, and map aligned ↔ unaligned positions.
  - Access a `DeepMutationalScanning` helper via `protein.dms`.

- Utility Functions (from `metrics.py` and `fasta.py`)
  - `compute_gap_frequency(msa_oh)`: gap fraction per column for a one-hot MSA.
  - `compute_seqID(a1, single_seq)`: sequence identity scores against a reference.
  - `count_gaps(msa)`: number of gaps per sequence (works for single or batched inputs).
  - `import_unaligned_fasta(...)`: read FASTA files with optional token filtering and duplicate removal.

## Testing

```bash
python -m unittest discover -s tests -v
```

## License

Proprietary / internal use only. Update this section if you plan to distribute the package.
