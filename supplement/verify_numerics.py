"""Independent operator checks and numerical diagnostics."""
from pathlib import Path
import json
import numpy as np
from scipy.sparse import csr_matrix,kron,eye
from physical_model import Hamiltonian,Flake,xy,geometry

DATA=Path(__file__).resolve().parent/'data'

def pauli_assembly(flake,D=2.,B=.65,hx=.2,hy=.07):
    s=[np.array([[0,1],[1,0]],complex)/2,
       np.array([[0,-1j],[1j,0]],complex)/2,
       np.diag([.5,-.5]).astype(complex)]
    ops=[];N=flake.N
    for i in range(N):
        site=[]
        for a in range(3):
            result=csr_matrix([[1.]])
            for q in reversed(range(N)):
                result=kron(result,csr_matrix(s[a]) if q==i else eye(2),format='csr')
            site.append(result)
        ops.append(site)
    H=csr_matrix((1<<N,1<<N),dtype=complex)
    for i,j,d in flake.bonds:
        ux,uy=xy(d);dx,dy=-D*uy,D*ux
        for a in range(3):H-=ops[i][a]@ops[j][a]
        H+=dx*(ops[i][1]@ops[j][2]-ops[i][2]@ops[j][1])
        H+=dy*(ops[i][2]@ops[j][0]-ops[i][0]@ops[j][2])
    for i,q,d in flake.boundary:
        ux,uy=xy(d);H+=-.5*ops[i][2]-.5*D*uy*ops[i][1]-.5*D*ux*ops[i][0]
    H-=B*sum(o[2] for o in ops)+hx*ops[0][0]+hy*ops[0][1]
    return H,ops

def verify(large=False):
    f=Flake(1);h=Hamiltonian(f,D=2.,hx=.2,hy=.07);H,ops=pauli_assembly(f)
    err=np.linalg.norm((H-h.matrix(.65)).toarray());assert err<1e-12
    e,v,r=h.solve(.65)
    independent=np.array([[np.vdot(v[:,0],op@v[:,0]).real for op in site] for site in ops])
    merr=np.max(abs(independent-h.moments(v[:,0])));assert merr<1e-13
    checks={'pauli_operator_frobenius_error':float(err),'local_moment_error':float(merr),'meshes':[]}
    for R in [1,2]:
        flake=Flake(R);inc={}
        for tri in flake.faces:
            for i,j in zip(tri,tri[1:]+tri[:1]):inc.setdefault(tuple(sorted((i,j))),[]).append((i,j))
        assert all(len(a)==2 and a[0]==a[1][::-1] for a in inc.values())
        chi=len(flake.mesh_sites)+1-len(inc)+len(flake.faces);assert chi==2
        checks['meshes'].append({'N':flake.N,'V':len(flake.mesh_sites)+1,'E':len(inc),'F':len(flake.faces),'Euler':chi})
    z=np.load(DATA/'n19_scan.npz');f=Flake(2);dualerr=0.
    for m in z['moments']:
        g=geometry(m,f);ext=f.extend(m)
        for tri,v in zip(f.faces,g['duals']):
            ids=[i for i in tri if f.weights[i]>0]
            if not ids:continue
            if any(f.weights[i]==0 for i in tri):assert v[2]>-1e-12
            from physical_model import face_clamped
            radius,q=face_clamped(ext[list(tri)],f.weights[list(tri)])
            lo=min(ext[i]@v for i in ids)
            dualerr=max(dualerr,abs(radius-lo))
    assert dualerr<1e-11;checks['clamped_primal_dual_error']=float(dualerr)
    if large:
        h=Hamiltonian(f,D=2.);rows=json.load(open(DATA/'n19_scan.json'));results=[]
        for ix in [0,4,12,15]:
            e,v,r=h.solve(rows[ix]['B'],k=6,seed=6031+ix,tol=5e-13)
            energy_error=abs(e[0]-rows[ix]['E0']);gap_error=abs(e[1]-e[0]-rows[ix]['gap'])
            assert energy_error<1e-8 and gap_error<1e-8
            results.append({'B':rows[ix]['B'],'E0_difference':float(energy_error),'gap_difference':float(gap_error),'max_residual':float(max(r))})
        checks['independent_large_starts']=results
    with open(DATA/'numerical_verification.json','w') as stream:json.dump(checks,stream,indent=2)
    print(json.dumps(checks,indent=2));return checks

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--large',action='store_true');verify(p.parse_args().large)
