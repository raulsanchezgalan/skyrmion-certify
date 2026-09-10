# Computational supplement

Companion to **Certified robustness of quantum spin textures under magnetic-field perturbations**.

The calculations use the full spin-1/2 Hilbert space for triangular quantum
flakes of 7 and 19 sites. A prescribed classical exterior is included through
linear boundary fields. The parameters are hbar = a = J = 1, D = 2 and K = 0.
The transverse-field example has (hx, hy) = (0.2, 0.07) at the central site.

## Files

- `quantum_skyrmion_revised.ipynb`: executable notebook with saved outputs.
- `physical_model.py`: lattice, linear Hamiltonian, local moments and geometry.
- `exact_audit.py`: exact integer/rational spectral, moment and degree verification.
- `verify_saved.py`: rechecks the saved candidates without an eigensolve.
- `run_analysis.py`: field scans, event localization and certificate construction.
- `verify_numerics.py`: independent Pauli assembly, mesh and eigensolver checks.
- `sensitivity_analysis.py`: field-refinement record, conditional error-budget
  propagation, and additional nineteen-spin solves with tighter tolerances.
- `make_figures.py`: regenerates the geometry schematic and the two figures of
  numerical results, as vector PDFs and PNG previews.
- `data/`: state vectors, moments, spectra, exact certificate inputs and bounds.

## Quick verification

From this directory:

```bash
python -m pip install -r requirements.txt
python verify_saved.py
python sensitivity_analysis.py
python make_figures.py
```

The first command installs the tested scientific packages. IPython and a notebook
frontend are additionally needed to run the notebook, but not the scripts.

`verify_saved.py` loads the quantized eigenbases and checks integer/rational
inequalities. It verifies all 128 eigenvalue enclosures, moment separating vectors,
exact signed-preimage degrees, the field-radius lower bound, and the uniform gap
and polarization bounds on the charge-changing interval. Floating-point arithmetic
can propose separating vectors, but their constraints and lower bounds are checked
exactly. No interval package is needed.

## Reproduce computations

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_analysis.py
python sensitivity_analysis.py
python make_figures.py
```

This recreates the seven-spin scans and exact certificates. The larger calculation
can be reproduced with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_analysis.py --large
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_numerics.py --large
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sensitivity_analysis.py --large
python make_figures.py
```

The nineteen-spin calculation uses several GB of memory. In the recorded
environment the 16 field values took several minutes. Independent-start checks
add further calculations. Its starting vectors include random components even
when a previous ground state is reused, avoiding confinement to a polarized
invariant subspace.

The analytical results apply to any spin s. The supplied numerical implementation
specializes to spin 1/2; it is not a general-spin simulation program. The general
field radius reduces algebraically to the expression used in these calculations.
The two available flake sizes are illustrations, not evidence for an asymptotic
size-scaling law. The paper proves the latter under explicit uniformity assumptions.

## Field refinement and nineteen-spin sensitivity

`data/field_refinement.json` records the scan step 0.0005 J, the adjacent
charge-change bracket [0.6480, 0.6485] J, the critical face (0,1,3), and a Brent
solve of its determinant. The verified interval [0.6480, 0.6490] J has convenient
rational endpoints and midpoint 0.6485 J. The midpoint audit controls the gap and
polarizations on both halves. The floating-point root is a localization, not an
exact enclosure of the root.

`data/n19_repeated_solves.json` compares the saved nineteen-spin results at
B/J = 0, 0.4, 1.4, 2 with fresh independent starts, six requested eigenpairs and
tolerance 2e-13. It retains moments, eigenvalues, residuals, seeds and discrepancies.
These comparisons do not bound a shared systematic error or rule out a missed
lower eigenvalue.

`data/n19_sensitivity.json` propagates the illustrative budgets max_i |delta m_i|
<= 1e-6 and |delta Delta| <= 1e-6 J through the Lipschitz geometric margin and
the monotone formula for R_B. These are conditional deterministic envelopes,
not verified input-error bounds or bootstrap confidence intervals. The script
accepts other budgets through `sensitivity_intervals(eps_m, eps_gap)`.

## Conventions and data

Site i is bit i; bit 0 means spin up. Positions are ordered by the hexagonal norm
max(|m|, |n|, |m+n|), then lexicographically. The seven-spin event face (0, 1, 3)
has axial coordinates (0,0), (-1,0), (0,-1). Every internal Hamiltonian bond is
counted once. Each exterior bond is oriented from its quantum endpoint to its
fixed classical endpoint.

The reconstruction mesh includes one classical shell and a constant cap. It is
an oriented sphere. Errors are allowed at quantum sites only: weights are one
there and zero at the exterior and cap.

The `.json` scan files contain energies, gaps, degree, both geometric radii,
polarizations, residuals and sufficient field radii. The `.npz` files contain
moments and spectra; seven-spin scans also contain the ground states. To keep the
archive small, the nineteen-spin file retains four selected ground states
(field-grid indices 0, 4, 12 and 15). All large-state moments and spectra are kept.

The degree is deliberately undefined at the numerical geometric event (`null`
in the event JSON). The existence of a singular reconstruction in the interval
is rigorously certified; the decimal root is a numerical localization.

The candidate matrices have real and imaginary components scaled by 2^40.
Hamiltonian entries are bounded via rational coefficients and an integer-square-root
enclosure of sqrt(3), then quantized with scale 2^48. See Appendix A for the proof
of the complete spectral enclosure.

Only the seven-spin audit and the stated interval implications are verified by
exact arithmetic. The nineteen-spin spectral ordering is supported by standard
floating-point eigensolver diagnostics and independent starts.

The notebook's six default code cells were executed in order in one IPython
process and its notebook structure was validated. The optional nineteen-spin
branch was executed separately for the saved data. Notebook outputs are supplied
for convenience; the scripts and exact checks are the reproducibility record.
