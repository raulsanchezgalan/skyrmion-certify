"""Deterministic sensitivity analysis; no claimed rigorous N=19 enclosure.

Default: propagate explicit hypothetical moment/gap error budgets and record
the seven-spin field bracket. --large additionally repeats four sparse solves
with independent starts and tighter tolerance, comparing moments and radii.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy.optimize import brentq
from physical_model import Hamiltonian, Flake, geometry, closest_hull, record

DATA = Path(__file__).resolve().parent / 'data'


def write(name, value):
    (DATA / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def field_bracket():
    rows = json.loads((DATA / 'n7_transverse.json').read_text())
    z = np.load(DATA / 'n7_transverse.npz')
    # Extra literal grid values can differ by one ulp from linspace values.
    grid = {}
    for k, row in enumerate(rows):
        grid[round(row['B'], 10)] = k
    fields = sorted(grid)
    changes = [(a, b) for a, b in zip(fields[:-1], fields[1:])
               if round(rows[grid[a]]['Q']) != round(rows[grid[b]]['Q'])]
    assert len(changes) == 1
    a, b = changes[0]
    assert a == .648 and b == .6485
    f = Flake(1)
    ia = geometry(z['moments'][grid[a]], f)['critical_face']
    ib = geometry(z['moments'][grid[b]], f)['critical_face']
    assert f.faces[ia] == f.faces[ib] == (0, 1, 3)
    tri = (0, 1, 3)
    h = Hamiltonian(f, D=2., hx=.2, hy=.07)

    def determinant(B):
        _, v, _ = h.solve(B)
        return float(np.linalg.det(h.moments(v[:, 0])[list(tri)]))

    root, info = brentq(determinant, a, b, xtol=5e-15, full_output=True)
    assert info.converged
    _, v, _ = h.solve(root)
    q, lam = closest_hull(h.moments(v[:, 0])[list(tri)])
    answer = {
        'scan_interval': [.6, .7], 'nominal_grid_step': .0005,
        'adjacent_charge_change_bracket': [a, b],
        'endpoint_charges': [round(rows[grid[a]]['Q']), round(rows[grid[b]]['Q'])],
        'minimum_radius_face_at_both_bracket_ends': list(tri),
        'determinants_at_bracket_ends': [determinant(a), determinant(b)],
        'refinement_method': 'scipy.optimize.brentq',
        'xtol': 5e-15, 'root': root, 'iterations': info.iterations,
        'function_calls': info.function_calls,
        'barycentric_weights_at_root': lam.tolist(),
        'face_margin_at_root': float(np.linalg.norm(q)),
        'verified_interval': [.648, .649], 'verified_midpoint': .6485,
        'reason_for_wider_verified_interval':
            'Convenient rational endpoints on opposite sides of the obstruction; '
            'the central certificate controls gap and moments over both half intervals.',
        'interpretation': 'Root localization is floating point; existence is certified separately.'
    }
    write('field_refinement.json', answer)
    return answer


def sensitivity_intervals(eps_m=1e-6, eps_gap=1e-6):
    """Conditional envelopes if the stated input error budgets are valid.

    This does not estimate or certify the unknown true eigensolver error.
    An independent-start discrepancy is empirical, not a confidence interval.
    """
    rows = json.loads((DATA / 'n19_scan.json').read_text())
    bounds = []
    for row in rows:
        g, d, p = row['Gamma'], row['gap'], row['p_min']
        gl, gu = max(0., g-eps_m), min(.5, g+eps_m)
        dl, du = max(0., d-eps_gap), d+eps_gap
        radius = [2*dl*gl/(19*(1+gl)), 2*du*gu/(19*(1+gu))]
        bounds.append({'B': row['B'], 'Gamma_interval': [gl, gu],
                      'gap_interval': [dl, du],
                      'p_min_interval': [max(0., p-2*eps_m), min(1., p+2*eps_m)],
                      'R_B_interval': radius,
                      'R_B_baseline': row['radius_B'],
                      'same_charge_under_moment_budget': bool(eps_m < g)})
    answer = {
        'kind': 'Conditional deterministic sensitivity envelopes, not verified eigensystem bounds',
        'moment_budget_max_site_Euclidean': eps_m,
        'gap_budget_in_J_units': eps_gap,
        'budget_selection': 'Round stress-test budgets, not statistical or interval error estimates',
        'assumptions': 'Correct ground-state identification and the stated gap/moment budgets',
        'minimum_sampled_Gamma': min(r['Gamma'] for r in rows),
        'minimum_sampled_gap': min(r['gap'] for r in rows),
        'maximum_R_B_change': max(max(abs(b-r['radius_B']) for b in v['R_B_interval'])
                                  for r, v in zip(rows, bounds)),
        'rows': bounds
    }
    write('n19_sensitivity.json', answer)
    return answer


def repeated_large_solves():
    rows = json.loads((DATA / 'n19_scan.json').read_text())
    saved = np.load(DATA / 'n19_scan.npz')
    f = Flake(2)
    h = Hamiltonian(f, D=2.)
    checks = []
    for ix in saved['selected_indices']:
        ix = int(ix)
        B = rows[ix]['B']
        start = time.monotonic()
        seed, tol, k = 12031+ix, 2e-13, 6
        e, v, r = h.solve(B, k=k, seed=seed, tol=tol)
        row, m = record(h, B, e, v, r)
        check = {
            'B': B, 'seed': seed, 'tolerance': tol, 'requested_eigenpairs': k,
            'reference_seed': 947+ix, 'reference_tolerance': 2e-12,
            'reference_requested_eigenpairs': 4,
            'E0_difference': abs(row['E0']-rows[ix]['E0']),
            'gap_difference': abs(row['gap']-rows[ix]['gap']),
            'max_site_moment_difference': float(np.linalg.norm(m-saved['moments'][ix],axis=1).max()),
            'Gamma_difference': abs(row['Gamma']-rows[ix]['Gamma']),
            'p_min_difference': abs(row['p_min']-rows[ix]['p_min']),
            'R_B_difference': abs(row['radius_B']-rows[ix]['radius_B']),
            'same_charge': round(row['Q']) == round(rows[ix]['Q']),
            'ground_residual': float(r[0]), 'max_residual': float(max(r)),
            'elapsed_seconds': time.monotonic()-start,
            'recomputed_moments': m.tolist(), 'recomputed_eigenvalues': e.tolist()
        }
        assert check['same_charge']
        assert check['max_site_moment_difference'] < 1e-8
        assert check['gap_difference'] < 1e-8
        checks.append(check)
        write('n19_repeated_solves.json', {'interpretation': 'Empirical solver agreement, not a certified error bound',
                                          'runs': checks})
        print(f'N=19 B={B:.1f}: moment difference={check["max_site_moment_difference"]:.3g}, '
              f'gap difference={check["gap_difference"]:.3g}, {check["elapsed_seconds"]:.1f}s', flush=True)
    return checks


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--large', action='store_true')
    args = p.parse_args()
    bracket = field_bracket()
    print('Refined numerical obstruction:', bracket['root'])
    sens = sensitivity_intervals()
    print('Conditional maximum field-radius change:', sens['maximum_R_B_change'])
    if args.large:
        repeated_large_solves()
