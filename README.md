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

msa.compute_gap_frequency()
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
- `SequencePath`: directed single-mutation paths between two aligned proteins,
  with greedy, flat-energy-step, and mean-energy constructors.
- `SequencePathFast`: the same path interface with precomputed single and
  pair-mutation effects for faster Monte Carlo sampling.
- FASTA and numerical helper functions.

```python
from adabmDCA_tools import SequencePath, SequencePathFast

vim2 = "..."
ndm1 = "..."

path = SequencePath.flat(
    vim2,
    ndm1,
    params,
    beta=1.0,
    steps=10_000,
    seed=7,
    keep_history=True,
)
print(path.mutations)        # 1-based directed mutation positions
print(path.score_history)    # objective before and after every MC step
path_msa = path.to_msa()     # VIM-2, every intermediate, and NDM-1
path.write_to_file("vim2_to_ndm1_path.fasta")

# Sample another path with the same settings.
path.make_path()

# Import an existing path from an ordered FASTA file.
path = SequencePath.from_file("vim2_to_ndm1_path.fasta", params)

# Use the same interface with faster flat and mean-energy sampling.
fast_path = SequencePathFast.flat(vim2, ndm1, params, steps=10_000)
```

The two wildtypes can be aligned sequence strings, `ProteinSequence` objects,
single-sequence FASTA paths, or vectors of encoded residues.

## Testing

```bash
python -m unittest discover -s tests -v
```

## License

Proprietary / internal use only.
