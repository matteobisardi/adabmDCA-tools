# adabmDCA_tools

Utilities for sequence-based analyses with [`adabmDCA`](https://github.com/spqb/adabmDCApy).
The package provides tools for multiple-sequence alignments, protein sequences,
FASTA files, PCA, and deep mutational scanning workflows.

## Installation

Install from GitHub:

```bash
python -m pip install git+https://github.com/matteobisardi/adabmDCA_tools.git
```

## Usage

```python
from adabmDCA_tools import MultipleSequenceAlignment, make_setup

setup = make_setup(alphabet="protein", device="cpu")
msa = MultipleSequenceAlignment.from_path(
    "alignment.fasta",
    setup=setup,
)

msa.compute_gap_frequency(th=0.8)
msa.compute_weights_cls()
msa.summary()
```

For unaligned FASTA files:

```python
from adabmDCA_tools import import_unaligned_fasta

headers, sequences = import_unaligned_fasta("sequences.fasta")
```

## Main components

- `MultipleSequenceAlignment`: aligned FASTA input, filtering, sequence weights, gap statistics, and PCA.
- `ProteinSequence`: aligned and unaligned protein sequences with position mapping.
- `DeepMutationalScanning`: DMS analysis through `protein.dms`.
- FASTA and numerical helper functions.

## Testing

```bash
python -m unittest discover -s tests -v
```

## License

Proprietary / internal use only.
