"""Spin-1/2 chiral Heisenberg flakes with a prescribed polarized exterior.

Units: hbar=a=J=1. Site i is bit i, with bit 0 denoting spin up.
No self-consistent fields or state-dependent Hamiltonian coefficients occur.
"""
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

DISPS = ((1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1))

def shell(p):
    x,y=p
    return max(abs(x),abs(y),abs(x+y))

def points(radius):
    return sorted(((x,y) for x in range(-radius,radius+1)
                   for y in range(-radius,radius+1) if shell((x,y))<=radius),
                  key=lambda p:(shell(p),p[0],p[1]))

def xy(p):
    return np.array([p[0]+p[1]/2,np.sqrt(3)*p[1]/2])

@dataclass
class Flake:
    radius: int = 1

    def __post_init__(self):
        self.sites=points(self.radius)
        self.N=len(self.sites)
        self.index={p:i for i,p in enumerate(self.sites)}
        self.positions=np.array([xy(p) for p in self.sites])
        self.bonds=[];self.boundary=[]
        for p,i in self.index.items():
            for d in DISPS:
                q=(p[0]+d[0],p[1]+d[1])
                if q in self.index:
                    j=self.index[q]
                    if i<j:self.bonds.append((i,j,d))
                else:self.boundary.append((i,q,d))
        # One prescribed classical shell, followed by a constant cap, gives S^2.
        self.mesh_sites=points(self.radius+1)
        idx={p:i for i,p in enumerate(self.mesh_sites)}
        faces=[]
        for x,y in self.mesh_sites:
            for tri in [((x,y),(x+1,y),(x,y+1)),
                        ((x+1,y+1),(x,y+1),(x+1,y))]:
                if all(p in idx for p in tri):faces.append(tuple(idx[p] for p in tri))
        edge_count={}
        for tri in faces:
            for i,j in zip(tri,tri[1:]+tri[:1]):
                key=tuple(sorted((i,j)))
                edge_count.setdefault(key,[]).append((i,j))
        cap=len(self.mesh_sites)
        for orientations in edge_count.values():
            if len(orientations)==1:
                i,j=orientations[0]
                faces.append((j,i,cap))
        self.faces=tuple(faces)
        self.mesh_positions=np.vstack([np.array([xy(p) for p in self.mesh_sites]),[np.nan,np.nan]])
        self.weights=np.array([1.]*self.N+[0.]*(len(self.mesh_sites)+1-self.N))

    def extend(self,m):
        out=np.tile([0.,0.,.5],(len(self.mesh_sites)+1,1))
        out[:self.N]=m
        return out


class Hamiltonian:
    """Sparse exact-Hilbert-space operator, assembled by bit transitions.

    H=-sum(SxSx+SySy)-Jz sum SzSz + D sum (z x e).(Si x Sj)
      -B sum Sz + hpin Sz_center - hx Sx_center - hy Sy_center
      + prescribed exterior.
    Jz=1+K; exterior magnetization=(0,0,1/2), using the same bond couplings.
    """
    def __init__(self,flake,D=2.,Jz=1.,hpin=0.,hx=0.,hy=0.):
        self.flake=flake;self.N=flake.N;self.dim=1<<self.N
        self.D=D;self.Jz=Jz;self.hpin=hpin;self.hx=hx;self.hy=hy
        self.basis=np.arange(self.dim,dtype=np.int32)
        self.sz=np.array([.5-((self.basis>>i)&1) for i in range(self.N)])
        self.mz=self.sz.sum(axis=0)
        diagonal=self.hpin*self.sz[0]
        flipcoeff=np.zeros((self.N,self.dim),complex)
        rowparts=[];colparts=[];dataparts=[]
        for i,j,d in flake.bonds:
            u=xy(d);si=self.sz[i];sj=self.sz[j]
            diagonal-=Jz*si*sj
            ix=self.basis[si!=sj]
            rowparts.append(ix);colparts.append(ix^(1<<i)^(1<<j))
            dataparts.append(np.full(len(ix),-.5,dtype=complex))
            flipcoeff[i]+=D*sj*(1j*u[1]*si-u[0]/2)
            flipcoeff[j]+=D*si*(-1j*u[1]*sj+u[0]/2)
        for i,q,d in flake.boundary:
            u=xy(d);si=self.sz[i]
            diagonal-=.5*Jz*si
            flipcoeff[i]+=D*(.5j*u[1]*si-u[0]/4)
        flipcoeff[0]-=hx/2
        flipcoeff[0]+=1j*hy*self.sz[0]
        for i in range(self.N):
            keep=np.abs(flipcoeff[i])>1e-15
            ix=self.basis[keep]
            rowparts.append(ix);colparts.append(ix^(1<<i));dataparts.append(flipcoeff[i,keep])
        rowparts.append(self.basis);colparts.append(self.basis);dataparts.append(diagonal.astype(complex))
        self.H0=coo_matrix((np.concatenate(dataparts),(np.concatenate(rowparts),np.concatenate(colparts))),
                           shape=(self.dim,self.dim)).tocsr()
        self.H0.eliminate_zeros()

    def matrix(self,B):return self.H0-diags(B*self.mz,format='csr')

    def moments(self,psi):
        probs=np.abs(psi)**2
        out=np.zeros((self.N,3))
        for i in range(self.N):
            prod=psi.conj()*psi[self.basis^(1<<i)]
            out[i]=[.5*prod.sum().real,(-1j*self.sz[i]*prod).sum().real,self.sz[i]@probs]
        return out

    def solve(self,B,k=4,seed=941,method='auto',v0=None,tol=1e-11):
        H=self.matrix(B)
        if method=='dense' or (method=='auto' and self.N<=10):
            vals,vecs=eigh(H.toarray(),subset_by_index=[0,k-1],driver='evr')
        else:
            rng=np.random.default_rng(seed)
            # Random components prevent invariant polarized starting vectors
            # from concealing lower eigenvalues in other symmetry sectors.
            random=rng.normal(size=self.dim)+1j*rng.normal(size=self.dim)
            random/=np.linalg.norm(random)
            start=random if v0 is None else v0+.1*random
            vals,vecs=eigsh(H,k=k,which='SA',v0=start,tol=tol,ncv=max(32,4*k),maxiter=10000)
            ix=np.argsort(vals);vals=vals[ix];vecs=vecs[:,ix]
        residuals=np.linalg.norm(H@vecs-vecs*vals,axis=0)
        return vals,vecs,residuals


def closest_hull(vectors):
    m=np.asarray(vectors); candidates=[(m[i],np.eye(len(m))[i]) for i in range(len(m))]
    for i,j in combinations(range(len(m)),2):
        e=m[j]-m[i]
        if e@e>1e-28:
            t=np.clip(-(m[i]@e)/(e@e),0,1)
            lam=np.zeros(len(m));lam[i]=1-t;lam[j]=t
            candidates.append((lam@m,lam))
    if len(m)==3:
        E=(m[1:]-m[0]).T
        if np.linalg.matrix_rank(E,tol=1e-13)==2:
            z=np.linalg.lstsq(E,-m[0],rcond=None)[0]
            if min(z)>=-1e-13 and sum(z)<=1+1e-13:
                lam=np.array([1-sum(z),*z]);lam=np.maximum(lam,0);lam/=sum(lam)
                candidates.append((lam@m,lam))
    return min(candidates,key=lambda x:x[0]@x[0])

def face_clamped(m,weights):
    """Exact active-set geometry for a face with a +z fixed exterior.

    Its relevant polyhedron is conv(movable moments)+cone(+z).
    The general weighted theorem is proved in the manuscript.
    """
    movable=m[np.asarray(weights)>0]
    if not len(movable):return np.inf,np.zeros(3)
    if len(movable)==len(m):
        q,_=closest_hull(movable)
    else:
        q,_=closest_hull(movable)
        candidates=[q]
        # Minimize the xy distance subject to the unshifted z <= 0.
        if len(movable)==1:
            if movable[0,2]<=0:candidates.append(np.r_[movable[0,:2],0.])
        else:
            a,b=movable;e=b-a;lo=0.;hi=1.
            if abs(e[2])<1e-15:
                feasible=a[2]<=0
            else:
                cross=-a[2]/e[2]
                if e[2]>0:hi=min(hi,cross)
                else:lo=max(lo,cross)
                feasible=lo<=hi
            if feasible:
                e2=e[:2]@e[:2]
                t=np.clip(-(a[:2]@e[:2])/e2,lo,hi) if e2>1e-28 else lo
                p=a+t*e;candidates.append(np.r_[p[:2],0.])
        q=min(candidates,key=lambda x:x@x)
    return float(np.linalg.norm(q)),q

def geometry(m,flake):
    extended=flake.extend(m);g=[];G=[];duals=[];omegas=[]
    norms=np.linalg.norm(extended,axis=1)
    for tri in flake.faces:
        f=extended[list(tri)];w=flake.weights[list(tri)]
        q,_=closest_hull(f);g.append(np.linalg.norm(q))
        radius,q=face_clamped(f,w);G.append(radius)
        duals.append(q/radius if np.isfinite(radius) and radius>1e-15 else np.zeros(3))
        if min(norms[list(tri)])<1e-13:omegas.append(np.nan);continue
        u,v,z=f/norms[list(tri),None]
        omegas.append(2*np.arctan2(u@np.cross(v,z),1+u@v+v@z+z@u))
    k=int(np.argmin(G))
    return {'gamma':float(min(g)),'Gamma':float(min(G)),
            'Q':float(sum(omegas)/(4*np.pi)) if min(G)>1e-10 else float('nan'),
            'p_min':float(2*np.min(norms[:flake.N])),
            'critical_face':k,'duals':np.array(duals)}

def record(model,B,vals,vecs,residuals):
    psi=vecs[:,0];m=model.moments(psi);g=geometry(m,model.flake)
    gap=float(vals[1]-vals[0]);Gamma=g['Gamma']
    return {'N':model.N,'D':model.D,'Jz':model.Jz,'B':float(B),'hx':model.hx,'hy':model.hy,
            'E0':float(vals[0]),'gap':gap,'max_residual':float(max(residuals)),
            'Q':g['Q'],'Gamma':Gamma,'gamma':g['gamma'],'p_min':g['p_min'],
            'm0z':float(m[0,2]),'radius_B':float(2*gap*Gamma/(model.N*(1+Gamma))),
            'polarized_overlap':float(abs(psi[0])**2)},m
