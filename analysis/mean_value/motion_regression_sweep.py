"""
Does adaptive motion regression actually remove more motion?

Static OLS leaves about 60% of the slow CLE-CRE unexplained, so the obvious
next move is to let the coefficients track: the mask re-seats, skin contact
changes, and the same head angle need not give the same capacitance at hour 7
as at hour 1. RLS with a forgetting factor does that.

The trap is that an adaptive filter with a short memory will absorb ANY slow
structure, including whatever physiology the signal carries. R^2 going up is
therefore not evidence that motion removal improved.

The control: run the identical filter against a SURROGATE regressor -- the same
head signal circularly shifted by half the night, so it has the same statistics
and the same step-like character but no true time alignment. Whatever the
surrogate explains is what the filter would have explained from nothing. The
difference between real and surrogate is the only part attributable to motion.

Swept both ways: the 3-axis gravity vector, and the 1-D head-turn angle.

Writes  reports/mean_value/rls_sweep_headangle.csv

Usage
-----
    .venv/Scripts/python.exe analysis/mean_value/motion_regression_sweep.py
"""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, importlib.util
spec=importlib.util.spec_from_file_location('m','analysis/mean_value/diff_motion_regressed.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from sleep_monitor.sessions import SESSION_META
from sleep_monitor import load_session
from sleep_monitor.filters import lowpass
from sleep_monitor.config import CAP_SCALE_TO_FF

LAMS=[0.9999,0.999,0.995,0.99]
rows=[]
for meta in SESSION_META:
    s=load_session(meta); n=int(m.BLOCK_S*m.FS)
    g={ax: lowpass(np.asarray(s.cap[ax],float), m.GRAV_HZ, m.FS) for ax in ('aX','aY','aZ')}
    gx,gy,gz=[m.blocks(g[a],n) for a in ('aX','aY','aZ')]
    turn=np.degrees(np.arctan2(gy,np.sqrt(gx**2+gz**2)))          # 1-D head angle
    cle=m.blocks(np.asarray(s.cap['CLE'],float)*CAP_SCALE_TO_FF,n)
    cre=m.blocks(np.asarray(s.cap['CRE'],float)*CAP_SCALE_TO_FF,n)
    d=(cle-cre); d=d-np.nanmean(d)
    T=turn.reshape(-1,1)
    Ts=np.roll(T,len(T)//2,axis=0)
    row={'session':meta['label']}
    _,res_s,r2_s,_=m._regress(d,T); row['static_1d']=r2_s
    _,_,r2_ss,_=m._regress(d,Ts);   row['static_1d_surr']=r2_ss
    for lam in LAMS:
        row[f'rls_{lam}']=m._r2(d,m.rls(d,T,lam))
        row[f'surr_{lam}']=m._r2(d,m.rls(d,Ts,lam))
    rows.append(row)
    print('  '+meta['label'], f"static {row['static_1d']:.2f} (surr {row['static_1d_surr']:.2f})",
          ' '.join(f"L{l} {row[f'rls_{l}']:.2f}/{row[f'surr_{l}']:.2f}" for l in LAMS))
t=pd.DataFrame(rows); t.to_csv('reports/mean_value/rls_sweep_headangle.csv',index=False)
print('\nmedian across 12 — 1-D head angle')
print(f"  static      real {t.static_1d.median():.2f}   shuffled {t.static_1d_surr.median():.2f}   genuine gain {t.static_1d.median()-t.static_1d_surr.median():+.2f}")
for lam in LAMS:
    a,b=t[f'rls_{lam}'].median(),t[f'surr_{lam}'].median()
    print(f"  RLS {lam}: real {a:.2f}   shuffled {b:.2f}   genuine gain {a-b:+.2f}")
