"""Compare reference runs with GFDL's sample output (Intel build, 2012).

Usage: python compare_sample.py [RUNROOT] [CASE ...]
Prints, per case and file, the worst relative difference over all variables
(normalised by the field's max |value|), and the variable where it occurs.
"""
import sys, os, glob
import numpy as np
import netCDF4

REF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REF, 'runs', 'strict')
cases = sys.argv[2:] or sorted(os.listdir(os.path.join(REF, 'sample_output')))
for case in cases:
    for f in sorted(glob.glob(os.path.join(REF, 'sample_output', case, 'history', '*.nc'))):
        b = os.path.basename(f)
        mine = os.path.join(root, case, b)
        if not os.path.exists(mine):
            print(f'{case:24s} {b:18s} MISSING'); continue
        a, m = netCDF4.Dataset(f), netCDF4.Dataset(mine)
        worst, wv, nvar = 0.0, '', 0
        for v in a.variables:
            if v not in m.variables or a.variables[v].ndim < 2: continue
            x = np.ma.filled(a.variables[v][:].astype(float), np.nan)
            y = np.ma.filled(m.variables[v][:].astype(float), np.nan)
            if x.shape != y.shape: print('shape', case, v, x.shape, y.shape); continue
            scale = np.nanmax(np.abs(x)) if np.isfinite(x).any() else 0
            if scale == 0 or not np.isfinite(scale): continue
            d = np.nanmax(np.abs(x - y)) / scale
            nvar += 1
            if d > worst: worst, wv = d, v
        print(f'{case:24s} {b:18s} nvar={nvar:3d} worst rel diff {worst:9.2e} ({wv})')
