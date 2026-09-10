"""Reproduce the paper's data. Use --large for the 19-spin calculation."""
from pathlib import Path
from fractions import Fraction as F
import argparse,json,platform,time
from datetime import datetime, timezone
import numpy as np
import scipy
from scipy.optimize import brentq
from physical_model import Hamiltonian,Flake,geometry,closest_hull,record
from exact_audit import audit

HERE=Path(__file__).resolve().parent
DATA=HERE/'data'

def dump(name,value):
    with open(DATA/name,'w') as stream:json.dump(value,stream,indent=2,allow_nan=False)

def scan_small():
    DATA.mkdir(exist_ok=True)
    for name,hx,hy,grid in [
        ('n7_scan',0.,0.,np.linspace(0,2,201)),
        ('n7_transverse',.2,.07,np.unique(np.r_[np.linspace(.60,.70,201),.648,.6485,.649]))]:
        h=Hamiltonian(Flake(1),D=2.,hx=hx,hy=hy);rows=[];moments=[];spectra=[];states=[]
        for B in grid:
            e,v,r=h.solve(B);row,m=record(h,B,e,v,r)
            rows.append(row);moments.append(m);spectra.append(e);states.append(v[:,0])
        dump(name+'.json',rows)
        np.savez_compressed(DATA/(name+'.npz'),B=grid,moments=moments,spectra=spectra,states=states)
    h=Hamiltonian(Flake(1),D=2.,hx=.2,hy=.07)
    tri=(0,1,3);fi=h.flake.faces.index(tri)
    def determinant(B):
        e,v,r=h.solve(B);return np.linalg.det(h.moments(v[:,0])[list(tri)])
    Bstar=brentq(determinant,.648,.649,xtol=5e-15)
    e,v,r=h.solve(Bstar);row,m=record(h,Bstar,e,v,r);row['Q']=None
    q,lam=closest_hull(m[list(tri)])
    n=m/np.linalg.norm(m,axis=1)[:,None];u,vv,w=n[list(tri)]
    amin=1+u@vv+vv@w+w@u
    other=[];ext=h.flake.extend(m)
    for f in h.flake.faces:
        if f!=tri:other.append(np.linalg.norm(closest_hull(ext[list(f)])[0]))
    out={'row':row,'face_index':fi,'face':list(tri),'barycentric_weights':lam.tolist(),
         'solid_angle_denominator':float(amin),'other_face_margin_min':float(min(other)),
         'determinant_derivative':float((determinant(Bstar+1e-6)-determinant(Bstar-1e-6))/2e-6)}
    dump('geometric_event.json',out)
    np.savez_compressed(DATA/'event_state.npz',B=Bstar,psi=v[:,0],moments=m,spectrum=e)
    # The zero-transverse-field level crossing is located in the subspace
    # orthogonal to the exact polarized eigenvector.
    from scipy.linalg import eigh
    h0=Hamiltonian(Flake(1),D=2.)
    def difference(B):
        H=h0.matrix(B).toarray()
        return eigh(H[1:,1:],subset_by_index=[0,0],driver='evr')[0][0]-H[0,0].real
    Bc=brentq(difference,.60,.70,xtol=5e-15)
    dump('symmetric_crossing.json',{'B_crossing':Bc})
    return out

def audits():
    results=[]
    for B,hx,hy in [('0.2','0','0'),('0.648','0.2','0.07'),('0.6485','0.2','0.07'),('0.649','0.2','0.07')]:
        result=audit(B=B,hx=hx,hy=hy,save_candidates=DATA/f'audit_B{B}_hx{hx}_hy{hy}.npz')
        results.append(result)
    dump('exact_audits.json',results)
    mid=results[2]['exact_bounds'];dl=F(mid['gap_lower']);pl=F(mid['polarization_lower']);eps=F(7,4000)
    interval={'left':'0.648','right':'0.649','gap_lower':str(dl-2*eps),
              'polarization_lower':str(pl-2*eps/(dl-eps))}
    assert F(interval['gap_lower'])>F('0.0672')
    assert F(interval['polarization_lower'])>F('0.442')
    assert results[1]['Q_certified']==-1 and results[3]['Q_certified']==0
    dump('interval_certificate.json',interval)
    return results

def scan_large():
    h=Hamiltonian(Flake(2),D=2.);rows=[];moments=[];spectra=[];selected=[];prev=None
    grid=[0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.,1.2,1.4,1.6,1.8,2.]
    selected_indices=[0,4,12,15]
    for k,B in enumerate(grid):
        start=time.time();e,v,r=h.solve(B,k=4,seed=947+k,v0=prev,tol=2e-12)
        row,m=record(h,B,e,v,r);rows.append(row);moments.append(m);spectra.append(e);prev=v[:,0]
        if k in selected_indices:selected.append(prev)
        print(f'N=19 B={B:.2f}: Q={row["Q"]:.0f}, gap={row["gap"]:.8f}, {time.time()-start:.1f}s',flush=True)
        dump('n19_scan.json',rows)
    np.savez_compressed(DATA/'n19_scan.npz',moments=moments,spectra=spectra,
                        selected_indices=selected_indices,selected_states=selected)
    return rows

def provenance():
    dump('provenance.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
        'date':datetime.now(timezone.utc).date().isoformat(),
        'original_data_date':'2026-09-09','units':'hbar = a = J = 1','D':2.,'K':0.,
        'model':'linear chiral Heisenberg Hamiltonian with prescribed +z polarized exterior',
        'small_exact_audit':'integer and rational arithmetic; candidate diagonalization is not trusted',
        'large_solver':'full Hilbert-space sparse eigsh; no interval ordering certificate for N=19',
        'large_seed':'947 + field-grid index','large_tolerance':2e-12,
        'large_sensitivity_seed':'12031 + field-grid index',
        'large_sensitivity_tolerance':2e-13,'large_sensitivity_eigenpairs':6,
        'spin':.5,'general_spin_results':'analytical only; scripts specialize to spin 1/2'})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--large',action='store_true')
    args=parser.parse_args();print(json.dumps(scan_small(),indent=2));audits()
    if args.large:scan_large()
    provenance()
