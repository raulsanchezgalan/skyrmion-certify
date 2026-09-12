"""Exact integer/rational certificates for the seven-spin numerical examples.

Floating-point diagonalization supplies candidates only. Bounds and the
oriented degree below are checked using Python integers and Fractions.
No interval or arbitrary-precision third-party package is required.
"""
from fractions import Fraction as F
from math import isqrt
import numpy as np
from scipy.linalg import eigh
from physical_model import Hamiltonian, Flake, geometry

def nearest(x):return (2*x.numerator+x.denominator)//(2*x.denominator)
def ceilroot(n):
    k=isqrt(int(n));return k if k*k==n else k+1
def frob_upper(a,b,den):
    return F(ceilroot(sum(int(x)**2+int(y)**2 for x,y in zip(a.flat,b.flat))),den)

def exact_matrix(flake,D=F(2),Jz=F(1),B=F(0),hpin=F(0),hx=F(0),hy=F(0)):
    """H=A+i (sqrt(3) C+T) with rational A,C,T."""
    d=1<<flake.N
    A=np.full((d,d),F(0),dtype=object);C=A.copy();T=A.copy()
    for row in range(d):
        s=[F(1-2*((row>>i)&1),2) for i in range(flake.N)]
        A[row,row]=-B*sum(s)+hpin*s[0]
        for i,j,(p,q) in flake.bonds:
            A[row,row]-=Jz*s[i]*s[j]
            if s[i]!=s[j]:A[row,row^(1<<i)^(1<<j)]-=F(1,2)
            ci=row^(1<<i);cj=row^(1<<j)
            A[row,ci]-=D*s[j]*F(2*p+q,4)
            A[row,cj]+=D*s[i]*F(2*p+q,4)
            C[row,ci]+=D*s[i]*s[j]*F(q,2)
            C[row,cj]-=D*s[i]*s[j]*F(q,2)
        for i,_,(p,q) in flake.boundary:
            A[row,row]-=Jz*s[i]/2
            ci=row^(1<<i)
            A[row,ci]-=D*F(2*p+q,8)
            C[row,ci]+=D*s[i]*F(q,4)
        A[row,row^1]-=hx/2
        T[row,row^1]+=hy*s[0]
    return A,C,T

def quantized_matrix(A,C,T,bits=48,root_bits=90):
    scale=1<<bits;rs=1<<root_bits;k=isqrt(3*rs*rs)
    assert k*k<3*rs*rs<(k+1)*(k+1)
    root_mid=F(2*k+1,2*rs)
    ar=np.array([[nearest(x*scale) for x in row] for row in A],dtype=object)
    ai=np.array([[nearest((x*root_mid+t)*scale) for x,t in zip(row,trow)] for row,trow in zip(C,T)],dtype=object)
    assert np.all(ar==ar.T) and np.all(ai==-ai.T)
    maxc=max(abs(x) for x in C.flat)
    # Each complex entry error <= |Re error|+|Im error|.
    err=len(A)*(F(1,scale)+maxc/F(2*rs))
    return ar,ai,scale,err

def exact_moments(x,y,N):
    n2=sum(int(a)**2+int(b)**2 for a,b in zip(x,y));nums=[]
    for i in range(N):
        ax=ay=az=0
        for k in range(len(x)):
            f=k^(1<<i);sgn=1-2*((k>>i)&1)
            ax+=int(x[k])*int(x[f])+int(y[k])*int(y[f])
            ay+=sgn*(int(x[k])*int(y[f])-int(y[k])*int(x[f]))
            az+=sgn*(int(x[k])**2+int(y[k])**2)
        nums.append([ax,ay,az])
    return np.array(nums,dtype=object),2*n2

def det(a,b,c):return sum(int(a[i])*(int(b[(i+1)%3])*int(c[(i+2)%3])-int(b[(i+2)%3])*int(c[(i+1)%3])) for i in range(3))
def cross(a,b):return [int(a[(i+1)%3])*int(b[(i+2)%3])-int(a[(i+2)%3])*int(b[(i+1)%3]) for i in range(3)]
def dot(a,b):return sum(int(x)*int(y) for x,y in zip(a,b))

def degree_integer(m,flake):
    """Degree by signed preimages of an exactly tested regular ray."""
    for u in [(137,223,317),(17,29,43),(101,197,389),(19,31,71)]:
        degree=0;regular=True
        for tri in flake.faces:
            a,b,c=m[list(tri)];delta=det(a,b,c)
            if delta==0:
                normals=[cross(a,b),cross(b,c),cross(c,a)]
                normal=next((n for n in normals if any(n)),None)
                if normal is None:
                    if not any(cross(a,u)):regular=False;break
                elif dot(normal,u)==0:regular=False;break
                continue
            signed=np.array([det(u,b,c),det(a,u,c),det(a,b,u)],dtype=object)*(1 if delta>0 else -1)
            if all(s>=0 for s in signed) and any(s==0 for s in signed):regular=False;break
            if all(s>0 for s in signed):degree+=1 if delta>0 else -1
        if regular:return degree,list(u)
    raise RuntimeError('Choose another regular ray.')

def certify(B='0.2',D='2',Jz='1',hx='0',hy='0',save_candidates=None,load_candidates=None):
    flake=Flake(1);d=1<<flake.N
    A,C,T=exact_matrix(flake,F(D),F(Jz),F(B),hx=F(hx),hy=F(hy))
    ar,ai,hs,herr=quantized_matrix(A,C,T)
    H=np.array(ar,dtype=float)/hs+1j*np.array(ai,dtype=float)/hs
    model=Hamiltonian(flake,D=float(D),Jz=float(Jz),hx=float(hx),hy=float(hy))
    assert np.max(np.abs(H-model.matrix(float(B)).toarray()))<2e-13
    xs=1<<40;ls=1<<40
    if load_candidates is None:
        ev,V=eigh(H,driver='evd')
        xr=np.rint(V.real*xs).astype(np.int64).astype(object)
        xi=np.rint(V.imag*xs).astype(np.int64).astype(object)
        lv=np.rint(ev*ls).astype(np.int64).astype(object)
    else:
        with np.load(load_candidates) as z:
            assert int(z['scale'])==xs
            xr=z['Xreal'].astype(object);xi=z['Ximag'].astype(object);lv=z['eigenvalues'].astype(object)
        assert xr.shape==(d,d) and xi.shape==(d,d) and lv.shape==(d,)
        assert all(lv[k]<=lv[k+1] for k in range(d-1))
    ga=xr.T@xr+xi.T@xi-np.eye(d,dtype=object)*xs**2
    gb=xr.T@xi-xi.T@xr
    delta=frob_upper(ga,gb,xs**2)
    assert delta<F(1)
    rr=ar@xr-ai@xi-xr*lv*(hs//ls)
    ri=ar@xi+ai@xr-xi*lv*(hs//ls)
    residual_matrix=frob_upper(rr,ri,hs*xs)
    hnorm=frob_upper(ar,ai,hs);dnorm=F(max(abs(int(v)) for v in lv),ls)
    enclosure=residual_matrix+(hnorm+dnorm)*delta+herr
    l0=F(int(lv[0]),ls);l1=F(int(lv[1]),ls)
    gap_lower=l1-l0-2*enclosure
    assert gap_lower>0
    normx_lower=F(isqrt(sum(int(a)**2+int(b)**2 for a,b in zip(xr[:,0],xi[:,0]))),xs)
    rvector=frob_upper(rr[:,0],ri[:,0],hs*xs)/normx_lower+herr
    eta=rvector/(l1-enclosure-l0)
    nums,den=exact_moments(xr[:,0],xi[:,0],flake.N)
    moment_norm_lower=min(F(isqrt(sum(int(z)**2 for z in row)),den) for row in nums)
    polarization_lower=2*(moment_norm_lower-eta)
    m=np.array(nums,dtype=float)/den
    full=np.vstack([nums,np.tile([0,0,den//2],(len(flake.mesh_sites)+1-flake.N,1))]).astype(object)
    g=geometry(m,flake);lower=[];dualnums=[]
    for tri,v in zip(flake.faces,g['duals']):
        w=flake.weights[list(tri)]
        if not np.any(w):dualnums.append([0,0,0]);continue
        vn=np.rint(v*(1-2**-28)*xs).astype(np.int64).astype(object)
        if np.any(w==0):vn[2]=max(0,vn[2])
        assert dot(vn,vn)<=xs**2
        if np.any(w==0):assert vn[2]>=0
        lo=min(F(dot(vn,full[i]),xs*den) for i in tri if flake.weights[i]>0)
        lower.append(lo);dualnums.append([int(x) for x in vn])
    Gamma_candidate=min(lower);Gamma_ground=Gamma_candidate-eta
    assert Gamma_ground>0
    Q,ray=degree_integer(full,flake)
    radius=2*gap_lower*Gamma_ground/(flake.N*(1+Gamma_ground))
    if save_candidates is not None:
        np.savez_compressed(save_candidates,Xreal=np.array(xr,dtype=np.int64),Ximag=np.array(xi,dtype=np.int64),
                            eigenvalues=np.array(lv,dtype=np.int64),scale=xs,duals=np.array(dualnums,dtype=np.int64))
    result={'N':7,'D':D,'Jz':Jz,'B':B,'hx':hx,'hy':hy,'Q_certified':Q,'regular_ray':ray,
            'spectrum_enclosure':float(enclosure),'gap_lower':float(gap_lower),
            'ground_state_trace_error_upper':float(eta),'Gamma_lower':float(Gamma_ground),
            'field_radius_lower':float(radius),'candidate_Gamma':g['Gamma'],
            'polarization_lower':float(polarization_lower),
            'orthogonality_error_upper':float(delta),'hamiltonian_rounding_error_upper':float(herr)}
    result['exact_bounds']={k:str(v) for k,v in [('spectrum_enclosure',enclosure),('gap_lower',gap_lower),
        ('trace_error_upper',eta),('Gamma_lower',Gamma_ground),('field_radius_lower',radius),
        ('polarization_lower',polarization_lower)]}
    return result

if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser();parser.add_argument('--B',default='0.2');parser.add_argument('--hx',default='0')
    args=parser.parse_args();print(json.dumps(certify(B=args.B,hx=args.hx),indent=2))
