# Certified robustness of quantum spin textures under magnetic-field perturbations

Code and computational supplement for:

> R. Sánchez Galán and R. Wieser, "Certified robustness of quantum spin textures
> under magnetic-field perturbations" (2026).
> Paper:
> Data/code archive DOI: 

This repository contains the exact-arithmetic verification code, numerical
data, and figure-generation scripts underlying the paper. The paper derives an
exact, computable distance ("radius") that a local spin-moment texture can be
perturbed by before its reconstructed topological charge becomes ill-defined,
and combines this with spectral perturbation theory to obtain certified
magnetic-field intervals over which a quantum ground-state texture's charge is
guaranteed to remain unchanged. The results are verified with exact integer
and rational arithmetic for a seven-spin chiral Heisenberg model, with a
larger nineteen-spin example provided as a floating-point illustration.

## Contents

```
figures/                  Vector (PDF) and raster (PNG) versions of the three paper figures
supplement/
  physical_model.py       Lattice, linear Hamiltonian, local moments, and mesh geometry
  exact_audit.py          Exact integer/rational spectral, moment, and degree verification
  verify_saved.py         Rechecks the saved certificates without re-running an eigensolver
  run_analysis.py         Field scans, event localization, and certificate construction
  verify_numerics.py      Independent Pauli-matrix assembly and eigensolver cross-checks
  sensitivity_analysis.py Field-refinement record and conditional error-budget propagation
  make_figures.py         Regenerates all three figures (PDF + PNG)
  quantum_skyrmion_revised.ipynb   Executable notebook with saved outputs
  data/                    Saved state vectors, moments, spectra, and certificate inputs
  requirements.txt
  README.md               Detailed notes on file conventions and data layout
```

## Quick verification

The fastest way to check the certified (exact-arithmetic) claims in the paper
without re-running any eigensolver:

```bash
cd supplement
python -m pip install -r requirements.txt
python verify_saved.py
```

`verify_saved.py` loads the quantized eigenbases saved in `data/` and checks,
using exact integer and rational arithmetic, all 128 eigenvalue enclosures for
the seven-spin Hamiltonian, the moment separating vectors, the exact
signed-preimage topological degree, the field-radius lower bound, and the
uniform spectral-gap and polarization bounds on the charge-changing interval
reported in the paper. Floating-point arithmetic is used only to *propose*
candidate eigenvectors and separating vectors; every inequality that
constitutes a certificate is checked exactly, with no interval-arithmetic
package required.

## Reproducing the computations from scratch

```bash
cd supplement
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_analysis.py
python sensitivity_analysis.py
python make_figures.py
```

This reproduces the seven-spin field scans and exact-arithmetic certificates.
The larger nineteen-spin calculation (full Hilbert-space sparse
diagonalization, dimension 524288) can be reproduced with the `--large` flag:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_analysis.py --large
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_numerics.py --large
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sensitivity_analysis.py --large
python make_figures.py
```

This uses several GB of memory and took several minutes per field value in
the original run. See `supplement/README.md` for full details on random
seeds, tolerances, and what is and is not covered by exact-arithmetic
enclosures.

## What is exact and what is not

- **Seven-spin results** (Section 5.1â5.2 of the paper) are backed by exact
  integer and rational arithmetic: the spectral ordering, the topological
  degree, the geometric admissibility radius, and the resulting field
  certificate are all proved correct by `exact_audit.py` /
  `verify_saved.py`, independent of floating-point rounding.
- **Nineteen-spin results** are conventional floating-point sparse
  diagonalization, checked by eigenpair residuals and independent random
  restarts, and are not accompanied by exact spectral-ordering enclosures.
  `sensitivity_analysis.py` provides conditional error-budget envelopes for
  these values; these are diagnostic, not certified bounds.
- All **analytical results in the paper** (the general spin-*s* radius, its
  sharpness, the size-scaling proposition) hold for arbitrary spin. The
  numerical implementation in this repository specializes to spin 1/2 and is
  not a general-spin simulation code.

## Requirements

```
numpy==2.3.5
scipy==1.17.0
matplotlib==3.10.8
```

Python 3.12 was used for the reference computations. A Jupyter/IPython
frontend is additionally required to run the notebook, but not the scripts.

## Citing

If you use this code or data, please cite the paper above. A citable
snapshot of this repository is also archived on Zenodo: .

## License

Released under the [MIT License](LICENSE).
