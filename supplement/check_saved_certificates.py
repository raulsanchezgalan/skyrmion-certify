"""Replay the stored spectral and geometry certificates without an eigensolve."""
from pathlib import Path
from fractions import Fraction as F
import json
from exact_certificates import certify

def verify():
    data=Path(__file__).resolve().parent/'data'
    rows=json.load(open(data/'exact_certificates.json'))
    for row in rows:
        B,hx,hy=row['B'],row['hx'],row['hy']
        candidate=data/f'cert_B{B}_hx{hx}_hy{hy}.npz'
        checked=certify(B=B,D=row['D'],Jz=row['Jz'],hx=hx,hy=hy,load_candidates=candidate)
        assert checked['Q_certified']==row['Q_certified']
        assert checked['exact_bounds']==row['exact_bounds']
        print(f'Verified B={B}, h_perp=({hx},{hy}): Q={checked["Q_certified"]}')
    interval=json.load(open(data/'interval_certificate.json'))
    mid=rows[2]['exact_bounds'];dl=F(mid['gap_lower']);pl=F(mid['polarization_lower']);eps=F(7,4000)
    assert F(interval['gap_lower'])==dl-2*eps>F('0.0672')
    assert F(interval['polarization_lower'])==pl-2*eps/(dl-eps)>F('0.442')
    assert rows[1]['Q_certified']==-1 and rows[3]['Q_certified']==0
    assert F(rows[0]['exact_bounds']['field_radius_lower'])>F('0.05992')
    print('All saved exact-arithmetic certificates and the interval implications verified.')
    return True

if __name__=='__main__':verify()
