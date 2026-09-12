#!/usr/bin/env python3
"""Exact replay of the manuscript's seven-spin certificates.

Default: python replay_certificates.py
Requires NumPy; SciPy is required only for --generate. No internet access is used.
All acceptance decisions use integers and fractions, never rounded JSON numbers.
The original manuscript and its implementation are not imported or modified.

The verification reconstructs the Hamiltonian from tensor-product Pauli operators.
The interval derivative bound and the event theorem are proved in the manuscript.
"""
from __future__ import annotations
import argparse
import json
from fractions import Fraction as F
from itertools import combinations
from math import isqrt, lcm
from pathlib import Path
from typing import Any
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def nearest(x: F) -> int:
    return (2*x.numerator+x.denominator)//(2*x.denominator)


def ceilroot(n: int) -> int:
    k = isqrt(n)
    return k + (k*k != n)


def frobenius(real: np.ndarray, imag: np.ndarray, denominator: int) -> F:
    n = sum(int(x)**2 + int(y)**2 for x, y in zip(real.flat, imag.flat))
    return F(ceilroot(n), denominator)


def det(a: Any, b: Any, c: Any) -> int:
    return sum(int(a[k])*(int(b[(k+1)%3])*int(c[(k+2)%3])
                          - int(b[(k+2)%3])*int(c[(k+1)%3])) for k in range(3))


def cross(a: Any, b: Any) -> tuple[int, int, int]:
    return tuple(int(a[(k+1)%3])*int(b[(k+2)%3])
                 - int(a[(k+2)%3])*int(b[(k+1)%3]) for k in range(3))


def dot(a: Any, b: Any) -> int:
    return sum(int(x)*int(y) for x, y in zip(a, b))


def shell(p: tuple[int, int]) -> int:
    return max(abs(p[0]), abs(p[1]), abs(p[0]+p[1]))


def lattice(radius: int) -> list[tuple[int, int]]:
    return sorted(((x,y) for x in range(-radius,radius+1)
                   for y in range(-radius,radius+1) if shell((x,y))<=radius),
                  key=lambda p: (shell(p), p[0], p[1]))


class SevenSpinModel:
    """Independent real-integer Pauli tensor assembly, with J=1, D=2, K=0."""
    def __init__(self) -> None:
        self.sites = lattice(1)
        self.mesh = lattice(2)
        self.N, self.d = 7, 128
        disps = ((1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1))
        index = {p:i for i,p in enumerate(self.sites)}
        self.bonds = []
        self.boundary = []
        for i,p in enumerate(self.sites):
            for dx,dy in disps:
                q = (p[0]+dx,p[1]+dy)
                if q not in index:
                    self.boundary.append((i,dx,dy))
                elif i < index[q]:
                    self.bonds.append((i,index[q],dx,dy))
        px = np.array([[0,1],[1,0]], dtype=np.int64)
        jy = np.array([[0,-1],[1,0]], dtype=np.int64)  # sigma_y = i*jy
        pz = np.diag([1,-1]).astype(np.int64)
        eye = np.eye(2,dtype=np.int64)
        self.ops = []
        for i in range(self.N):
            site = []
            for small in (px,jy,pz):
                op = np.ones((1,1),dtype=np.int64)
                for j in reversed(range(self.N)):
                    op = np.kron(op,small if j==i else eye)
                site.append(op)
            self.ops.append(site)
        self.A0 = np.zeros((self.d,self.d),dtype=np.int64)
        self.C = self.A0.copy()
        for i,j,p,q in self.bonds:
            x,y,z = self.ops[i]
            X,Y,Z = self.ops[j]
            self.A0 -= 50*(x@X-y@Y+z@Z)
            self.A0 += 50*(2*p+q)*(z@X-x@Z)
            self.C -= 50*q*(y@Z-z@Y)
        for i,p,q in self.boundary:
            x,y,z = self.ops[i]
            self.A0 -= 50*z + 50*(2*p+q)*x
            self.C -= 50*q*y
        self.Z = sum(ops[2] for ops in self.ops)
        # H=A0/200 - B*Z/2 - hx*X0/2 + i*(sqrt(3)*C/200 - hy*J0/2).
        idx = {p:i for i,p in enumerate(self.mesh)}
        faces = []
        for x,y in self.mesh:
            for tri in (((x,y),(x+1,y),(x,y+1)),
                        ((x+1,y+1),(x,y+1),(x+1,y))):
                if all(p in idx for p in tri):
                    faces.append(tuple(idx[p] for p in tri))
        edge_inc = {}
        for a,b,c in faces:
            for i,j in ((a,b),(b,c),(c,a)):
                edge_inc.setdefault(tuple(sorted((i,j))),[]).append((i,j))
        cap = len(self.mesh)
        for orientations in edge_inc.values():
            if len(orientations)==1:
                i,j = orientations[0]
                faces.append((j,i,cap))
        self.faces = tuple(faces)
        # Independent combinatorial check: all elementary triangles are present.
        elementary = set()
        for tri in combinations(range(len(self.mesh)),3):
            pts = [self.mesh[k] for k in tri]
            if all((a[0]-b[0])**2+(a[0]-b[0])*(a[1]-b[1])+(a[1]-b[1])**2==1
                   for a,b in combinations(pts,2)):
                elementary.add(frozenset(tri))
        legacy_disk = [t for t in self.faces if cap not in t]
        missing = elementary - {frozenset(t) for t in legacy_disk}
        require(len(missing)==2, 'Unexpected legacy mesh discrepancy')
        corrected_disk = legacy_disk.copy()
        for vertex_set in sorted(missing,key=lambda t: tuple(sorted(t))):
            tri = tuple(sorted(vertex_set))
            require(sum(i<7 for i in tri)==1, 'Missing face is not a single-variable face')
            a,b,c = (self.mesh[i] for i in tri)
            area = (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
            corrected_disk.append(tri if area>0 else (tri[0],tri[2],tri[1]))
        incidence = {}
        for a,b,c in corrected_disk:
            for i,j in ((a,b),(b,c),(c,a)):
                incidence.setdefault(tuple(sorted((i,j))),[]).append((i,j))
        self.corrected_faces = tuple(corrected_disk + [(v[0][1],v[0][0],cap)
                                      for v in incidence.values() if len(v)==1])
        require(all(all(i>=7 for i in t) for t in self.corrected_faces if cap in t),
                'Corrected cap is not wholly classical')
        require({frozenset(t) for t in self.corrected_faces if cap not in t}==elementary,
                'Incorrect corrected triangle inventory')
        def signatures(triangles):
            return {(tuple(sorted(i for i in t if i<7)),any(i>=7 for i in t)) for t in triangles}
        require(signatures(self.faces)==signatures(self.corrected_faces),
                'Geometric constraints differ after mesh repair')
        require({t for t in self.faces if sum(i<7 for i in t)>=2}
                =={t for t in self.corrected_faces if sum(i<7 for i in t)>=2},
                'Potentially nondegenerate oriented face changed')
        self.check_mesh()

    def check_mesh(self) -> None:
        inc, links = {}, {i:{} for i in range(len(self.mesh)+1)}
        for a,b,c in self.faces:
            for i,j,k in ((a,b,c),(b,c,a),(c,a,b)):
                inc.setdefault(tuple(sorted((i,j))),[]).append((i,j))
                links[i].setdefault(j,set()).add(k)
                links[i].setdefault(k,set()).add(j)
        require(all(len(v)==2 and v[0]==v[1][::-1] for v in inc.values()),
                'Mesh edge incidence/orientation failure')
        for link in links.values():
            require(all(len(v)==2 for v in link.values()),'Non-manifold vertex link')
            seen, todo = set(), [next(iter(link))]
            while todo:
                j=todo.pop()
                if j not in seen:
                    seen.add(j); todo.extend(link[j]-seen)
            require(len(seen)==len(link),'Disconnected vertex link')
        require(len(self.mesh)+1-len(inc)+len(self.faces)==2,'Wrong Euler characteristic')

    def quantized(self, B: F, hx: F, hy: F, hbits: int, rootbits: int):
        hs, rs = 1<<hbits, 1<<rootbits
        den = lcm(200, 2*B.denominator, 2*hx.denominator)
        anum = (self.A0.astype(object)*(den//200)
                - B.numerator*(den//(2*B.denominator))*self.Z.astype(object)
                - hx.numerator*(den//(2*hx.denominator))*self.ops[0][0].astype(object))
        k = isqrt(3*rs*rs)
        require(k*k < 3*rs*rs < (k+1)*(k+1),'Invalid sqrt(3) enclosure')
        mid = F(2*k+1,2*rs)
        ar = np.array([[nearest(F(int(x)*hs,den)) for x in row] for row in anum],dtype=object)
        ai = np.array([[nearest((F(int(self.C[i,j]),200)*mid
                                  -hy*F(int(self.ops[0][1][i,j]),2))*hs)
                        for j in range(self.d)] for i in range(self.d)],dtype=object)
        require(np.array_equal(ar,ar.T) and np.array_equal(ai,-ai.T),'Rounded matrix is not Hermitian')
        herr = self.d*(F(1,hs)+F(int(np.max(np.abs(self.C))),200)/(2*rs))
        return ar,ai,hs,herr

    def moments(self, xr: np.ndarray, xi: np.ndarray):
        xr,xi = xr.astype(object),xi.astype(object)
        n2 = int(xr@xr+xi@xi)
        require(n2>0,'Zero candidate vector')
        nums=[]
        for x,y,z in self.ops:
            X,Y,Z=x.astype(object),y.astype(object),z.astype(object)
            nums.append([int(xr@X@xr+xi@X@xi),
                         int(xi@Y@xr-xr@Y@xi),
                         int(xr@Z@xr+xi@Z@xi)])
        den=2*n2
        for row in nums:
            require(4*dot(row,row)<=den*den,'Unphysical exact candidate moment')
        full=np.array(nums+[[0,0,n2]]*(len(self.mesh)+1-self.N),dtype=object)
        return full,den

    def degree(self, full: np.ndarray, corrected: bool = False) -> int:
        u=(137,223,317)
        degree=0
        for tri in (self.corrected_faces if corrected else self.faces):
            a,b,c=full[list(tri)]
            d=det(a,b,c)
            if d==0:
                normals=[cross(a,b),cross(a,c),cross(b,c)]
                normal=next((n for n in normals if any(n)),None)
                if normal is not None:
                    require(dot(normal,u)!=0,'Ray may meet a rank-two face')
                else:
                    require(any(cross(a,u)),'Ray may meet a rank-one face')
                continue
            cs=[det(u,b,c),det(a,u,c),det(a,b,u)]
            sg=1 if d>0 else -1
            signed=[sg*x for x in cs]
            require(not(all(x>=0 for x in signed) and any(x==0 for x in signed)),
                    'Ray may meet a face edge')
            if all(x>0 for x in signed):
                degree+=sg
        return degree


def verify_candidate(model: SevenSpinModel, config: dict[str,Any]) -> dict[str,Any]:
    ar,ai,hs,herr=model.quantized(F(config['B']),F(config['hx']),F(config['hy']),
                                 config['hbits'],config['rootbits'])
    with np.load(DATA/config['file'],allow_pickle=False) as z:
        for key in ('Xreal','Ximag','eigenvalues','scale'):
            require(np.issubdtype(z[key].dtype,np.integer), f'Non-integer stored array: {key}')
        if 'duals' in z.files:
            require(np.issubdtype(z['duals'].dtype,np.integer), 'Non-integer stored duals')
        xr=z['Xreal'].astype(object); xi=z['Ximag'].astype(object)
        lv=z['eigenvalues'].astype(object); xs=int(z['scale'])
        duals=z['duals'].astype(object) if 'duals' in z.files else None
    require(xr.shape==xi.shape==(128,128) and lv.shape==(128,), 'Wrong candidate dimensions')
    require(xs>0 and (xs&(xs-1))==0 and hs%xs==0,'Unsupported exact scales')
    require(all(lv[i]<=lv[i+1] for i in range(127)),'Unordered candidate eigenvalues')
    ident=np.eye(128,dtype=object)
    ga=xr.T@xr+xi.T@xi-xs**2*ident
    gi=xr.T@xi-xi.T@xr
    delta=frobenius(ga,gi,xs**2)
    require(delta<1,'Candidate matrix may be singular')
    rr=ar@xr-ai@xi-xr*lv*(hs//xs)
    ri=ar@xi+ai@xr-xi*lv*(hs//xs)
    rmat=frobenius(rr,ri,hs*xs)
    hnorm=frobenius(ar,ai,hs)
    lnorm=F(max(abs(int(v)) for v in lv),xs)
    enclosure=herr+rmat+(hnorm+lnorm)*delta
    l0,l1=F(int(lv[0]),xs),F(int(lv[1]),xs)
    gap=l1-l0-2*enclosure
    require(gap>0,'Ground-state ordering not established')
    xnormlo=F(isqrt(sum(int(a)**2+int(b)**2 for a,b in zip(xr[:,0],xi[:,0]))),xs)
    require(xnormlo>0,'No positive candidate norm bound')
    eta=(frobenius(rr[:,0],ri[:,0],hs*xs)/xnormlo+herr)/(l1-enclosure-l0)
    full,den=model.moments(xr[:,0],xi[:,0])
    dc=F(det(*full[[0,1,3]]),den**3)
    derr=F(3,4)*eta
    pl=2*(min(F(isqrt(dot(row,row)),den) for row in full[:7])-eta)
    result={'B':config['B'],'hx':config['hx'],'hy':config['hy'],
            'spectrum_enclosure':enclosure,'gap_lower':gap,'trace_error_upper':eta,
            'polarization_lower':pl,'determinant_lower':dc-derr,
            'determinant_upper':dc+derr,'moment_numerators':full,'moment_denominator':den}
    if duals is not None:
        require(duals.shape==(len(model.faces),3),'Wrong dual dimensions')
        ds=xs
        margins={}
        for tri,v in zip(model.faces,duals):
            ids=[i for i in tri if i<7]
            if not ids:continue
            require(dot(v,v)<=ds**2,'Invalid dual norm')
            if len(ids)<3:require(v[2]>=0,'Invalid fixed-exterior dual constraint')
            margins[tri]=min(F(dot(v,full[i]),ds*den) for i in ids)
        gamma=min(margins.values())-eta
        require(gamma>0,'Candidate admissibility or residual transfer not established')
        require(model.degree(full)==model.degree(full,corrected=True), 'Degree changed after mesh correction')
        result.update(Gamma_lower=gamma,field_radius_lower=2*gap*gamma/(7*(1+gamma)),
                      Q_certified=model.degree(full),face_candidate_margins=margins)
    if 'expected_exact_bounds' in config:
        for key,value in config['expected_exact_bounds'].items():
            require(result[key]==F(value),f'Exact bound mismatch: {config["file"]}: {key}')
        require(result['Q_certified']==config['expected_Q'],'Wrong original degree')
    return result


def generate(model: SevenSpinModel, config: dict[str,Any]) -> None:
    from scipy.linalg import eigh
    ar,ai,hs,_=model.quantized(F(config['B']),F(config['hx']),F(config['hy']),
                              config['hbits'],config['rootbits'])
    H=np.asarray(ar,dtype=float)/hs+1j*np.asarray(ai,dtype=float)/hs
    vals,vecs=eigh(H,driver='evd')
    xs=1<<config['xbits']
    np.savez_compressed(DATA/config['file'],
                        Xreal=np.rint(vecs.real*xs).astype(np.int64),
                        Ximag=np.rint(vecs.imag*xs).astype(np.int64),
                        eigenvalues=np.rint(vals*xs).astype(np.int64),scale=xs)


def interval_conclusions(results: dict[str,dict[str,Any]]) -> dict[str,Any]:
    mid=results['original_midpoint']
    eps=F(7,4000)
    gap=mid['gap_lower']-2*eps
    pol=mid['polarization_lower']-2*eps/(mid['gap_lower']-eps)
    require(gap>F('0.0672') and pol>F('0.442'),'Uniform interval bounds failed')
    require(results['original_left']['Q_certified']==-1 and results['original_right']['Q_certified']==0,
            'Endpoint degrees failed')
    other=min(v for face,v in mid['face_candidate_margins'].items() if face!=(0,1,3))
    other-=mid['trace_error_upper']+eps/(mid['gap_lower']-eps)
    require(other>F('0.0155'),'Other-face uniform admissibility not proved')
    grid=[F('0.648')+i*F('0.0001') for i in range(11)]
    by_field={F(r['B']):r for r in results.values() if r['hx']=='0.2'}
    M=60*F(1,2)**3*F(7,2)**2/F('0.0672')**2
    slopes=[]
    for a,b in zip(grid[:-1],grid[1:]):
        lo=(by_field[b]['determinant_lower']-by_field[a]['determinant_upper'])/(b-a)-M*(b-a)/2
        require(lo>F('1.747'),f'Determinant derivative bound not established on {a}, {b}')
        slopes.append({'left':str(a),'right':str(b),'derivative_lower':str(lo),
                       'derivative_lower_decimal':float(lo)})
    left,right=results['tight_left'],results['tight_right']
    require(F('-3.452e-12')<=left['determinant_lower']<=left['determinant_upper']<=F('-3.148e-12'), 'Left display enclosure not established')
    require(F('2.261e-12')<=right['determinant_lower']<=right['determinant_upper']<=F('2.546e-12'), 'Right display enclosure not established')
    require(left['determinant_upper']<0<right['determinant_lower'],'Tight root signs not established')
    ref=results['original_reference']
    G=ref['Gamma_lower']; D=ref['gap_lower']; improved=F('0.07451')
    require(0<G<=F(1,2), 'Reference margin outside physical range')
    improved_squared=(2*D*G/7)**2*(1-G**2)
    require(improved_squared>improved**2, 'Improved reference radius not certified')
    return {'sharper_reference_radius_certified':str(improved),
            'sharper_reference_radius_squared_lower':str(improved_squared),
            'uniform_gap_lower':str(gap),'uniform_polarization_lower':str(pol),
            'other_face_radius_lower':str(other),'other_face_radius_lower_decimal':float(other),
            'determinant_second_derivative_bound':str(M),
            'determinant_derivative_lower':str(min(F(s['derivative_lower']) for s in slopes)),
            'determinant_derivative_lower_decimal':float(min(F(s['derivative_lower']) for s in slopes)),
            'unique_root_bracket':[left['B'],right['B']],
            'left_determinant_interval':[str(left['determinant_lower']),str(left['determinant_upper'])],
            'right_determinant_interval':[str(right['determinant_lower']),str(right['determinant_upper'])],
            'cell_certificates':slopes,
            'conclusion':'Exactly one reconstruction obstruction in [0.648,0.649]; '
                         'it is a transverse interior event on oriented face (0,1,3), with degree jump +1.'}


def json_safe_result(result: dict[str,Any]) -> dict[str,Any]:
    skip={'moment_numerators','moment_denominator','face_candidate_margins'}
    return {k:(str(v) if isinstance(v,F) else v) for k,v in result.items() if k not in skip}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',nargs='*',metavar='ID',
                        help='Generate selected new candidates (or all new candidates when no IDs follow)')
    parser.add_argument('--only',nargs='+',metavar='ID',help='Replay selected candidates only')
    args=parser.parse_args()
    manifest=json.loads((DATA/'manifest.json').read_text())
    model=SevenSpinModel()
    configs=manifest['certificates']
    known={c['id'] for c in configs}
    if args.only is not None: require(set(args.only)<=known, 'Unknown --only certificate ID')
    if args.generate is not None: require(set(args.generate)<=known, 'Unknown --generate certificate ID')
    if args.generate is not None:
        selected=set(args.generate) if args.generate else {c['id'] for c in configs if not c.get('original')}
        for c in configs:
            if c['id'] in selected:
                require(not c.get('original'),'Refusing to overwrite an original candidate')
                generate(model,c)
                print('Generated candidate',c['id'],flush=True)
    selected=set(args.only) if args.only else None
    results={}
    for c in configs:
        if selected is not None and c['id'] not in selected:continue
        r=verify_candidate(model,c);results[c['id']]=r
        print('Verified',c['id'],'B='+c['B'],
              'det in [%.12g, %.12g]'%(float(r['determinant_lower']),float(r['determinant_upper'])),
              'eta <= %.3g'%float(r['trace_error_upper']),flush=True)
    if selected is None:
        conclusion=interval_conclusions(results)
        payload={'verification_arithmetic':'Python integers and fractions',
                 'candidates':{k:json_safe_result(v) for k,v in results.items()},
                 'interval_conclusion':conclusion}
        (DATA/'certificate_report.json').write_text(json.dumps(payload,indent=2)+'\n')
        print(json.dumps(conclusion,indent=2),flush=True)
        print('All exact bounds and all certificates passed.',flush=True)


if __name__=='__main__':
    main()
