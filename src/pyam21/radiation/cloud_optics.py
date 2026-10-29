"""Cloud optical properties (sea_esf_rad/microphys_rad.F90, rad_utilities.F90
thickavg, siena_201207).

Shortwave: Slingo (1989) droplets and Fu & Liou (1993) ice (icesolar) on their own
spectral intervals, averaged onto the ESF bands with the TOA solar flux as weight
(thickavg: extinction averaged directly, single-scattering albedo through the
reflectance of a thick layer), then summed over the species (cloudpar). Rain and snow
are absent in this configuration (zero concentrations, as the Fortran's masks give).
Longwave: see cloud_lw.
"""
from __future__ import annotations

import numpy as np
import torch

from .esfsw_data import SolarSpectrum, band_intervals
from ..core import ssqrt

ENDLIQCLDWVN = [2924, 3437, 4202, 4695, 6098, 6536, 7813, 8404, 9091, 10000, 11494, 12821, 13333, 14493,
                15625, 17544, 19231, 20833, 22727, 25000, 27778, 30303, 33333, 57600]
ENDICESOLCLDWVN = [2857, 4000, 5263, 7692, 14493, 57600]

SLINGO_A = [-1.023, 1.950, 1.579, 1.850, 1.970, 2.237, 2.463, 2.551, 2.589, 2.632, 2.497, 2.622, 2.650, 3.115,
            2.895, 2.831, 2.838, 2.672, 2.698, 2.668, 2.801, 3.308, 2.944, 3.094]
SLINGO_B = [1.933, 1.540, 1.611, 1.556, 1.501, 1.452, 1.420, 1.401, 1.385, 1.365, 1.376, 1.362, 1.349, 1.244,
            1.315, 1.317, 1.300, 1.320, 1.315, 1.307, 1.293, 1.246, 1.270, 1.252]
SLINGO_C = [2.500e-02, 4.490e-01, 1.230e-01, 1.900e-04, 1.200e-03, 1.200e-04, 2.400e-04, 6.200e-05, -2.800e-05,
            -4.600e-05, 9.800e-06, 3.300e-06, 2.300e-06, -2.700e-07, -1.200e-07, -1.200e-06, 0.0, 0.0, 1.000e-06,
            0.0, 1.000e-06, -3.000e-07, -6.500e-07, 7.900e-07]
SLINGO_D = [1.220e-02, 1.540e-03, 9.350e-03, 2.540e-03, 2.160e-03, 6.670e-04, 8.560e-04, 2.600e-04, 8.000e-05,
            5.000e-05, 2.100e-05, 2.800e-06, 1.700e-06, 1.400e-06, 4.400e-07, 4.000e-07, 0.0, 0.0, 0.0, 0.0, 0.0,
            2.360e-07, 4.330e-07, 3.690e-07]
SLINGO_E = [7.260e-01, 8.310e-01, 8.510e-01, 7.690e-01, 7.400e-01, 7.490e-01, 7.540e-01, 7.730e-01, 7.800e-01,
            7.840e-01, 7.830e-01, 8.060e-01, 8.090e-01, 8.040e-01, 8.180e-01, 8.280e-01, 8.250e-01, 8.280e-01,
            8.200e-01, 8.400e-01, 8.360e-01, 8.390e-01, 8.410e-01, 8.440e-01]
SLINGO_F = [6.652, 6.102, 2.814, 5.171, 7.469, 6.931, 6.555, 5.405, 4.989, 4.745, 5.035, 3.355, 3.387, 3.520,
            2.989, 2.492, 2.776, 2.467, 3.004, 1.881, 2.153, 1.946, 1.680, 1.558]

# icesolar tables, Fortran b(1:6, 0:3) in storage order
_ICE_B = [.10998e-5, .20208e-4, .1359e-3, -.16598e-2, .4618, .42362e-1,
          -.26101e-7, .96483e-5, .73453e-3, .20933e-2, .24471e-3, .86425e-2,
          .10896e-8, .83009e-7, .28281e-5, -.13977e-5, -.27839e-5, -.75519e-4,
          -.47387e-11, -.32217e-9, -.18272e-7, -.18703e-7, .10379e-7, .24056e-6]
_ICE_C = [2.211, 2.2151, 2.2376, 2.3012, 2.7975, 1.9655,
          -.10398e-2, -.77982e-3, .10293e-2, .33854e-2, .29741e-2, .20094e-1,
          .65199e-4, .6375e-4, .50842e-4, .23528e-4, -.32344e-4, -.17067e-3,
          -.34498e-6, -.34466e-6, -.30135e-6, -.20068e-6, .11636e-6, .50806e-6]
_ICE_D = [.12495, .12363, .12117, .11581, -.15968e-3, .1383,
          -.43582e-3, -.44419e-3, -.48474e-3, -.55031e-3, .10115e-4, -.18921e-2,
          .14092e-4, .14038e-4, .12495e-4, .98776e-5, -.12472e-6, .1203e-4,
          -.69565e-7, -.68851e-7, -.62411e-7, -.50193e-7, .48667e-9, -.31698e-7]


def _tiny(x):
    """The Fortran's 1.0e-100 guard; in float32 that is 0, so use the smallest normal."""
    return 1.0e-100 if x.dtype == torch.float64 else torch.finfo(x.dtype).tiny


def _tab(lst):
    return np.array(lst).reshape(4, 6)          # [power, 1..6]


def thickavg(n1, n2, solivl, solflxband, ext, ssalb, asym, mask):
    """thickavg_3d: ext/ssalb/asym (ncol, k, nivl) -> band values (ncol, k, nbands)."""
    zero = torch.zeros_like(ext[..., 0])
    eb, sb, ab = [], [], []
    for b in range(len(n1)):
        sumk = zero; somk = zero; somkg = zero; sref = zero
        for ni in range(n1[b], n2[b] + 1):
            w = float(solivl[b, ni])
            om = torch.clamp(ssalb[..., ni], max=1.0)
            g = asym[..., ni]
            den = torch.where(mask, 1.0 - om * g, torch.ones_like(om))
            sp = ssqrt(torch.where(mask, (1.0 - om) / den, torch.ones_like(om)))
            refthick = (1.0 - sp) / (1.0 + sp)
            sref = torch.where(mask, sref + refthick * w, sref)
            sumk = torch.where(mask, sumk + ext[..., ni] * w, sumk)
            somk = torch.where(mask, somk + om * ext[..., ni] * w, somk)
            somkg = torch.where(mask, somkg + om * ext[..., ni] * g * w, somkg)
        e = sumk / solflxband[b]
        a = somkg / (somk + _tiny(somk))
        r = sref / solflxband[b]
        s = 4.0 * r / ((1.0 + r) ** 2 - a * (1.0 - r) ** 2)
        eb.append(e); sb.append(s); ab.append(a)
    return torch.stack(eb, -1), torch.stack(sb, -1), torch.stack(ab, -1)


class CloudSWOptics:
    def __init__(self, spec: SolarSpectrum, min_drop=4.2, max_drop=16.6, min_ice=18.6, max_ice=130.2):
        self.spec = spec
        self.liq = band_intervals(spec, ENDLIQCLDWVN)
        self.ice = band_intervals(spec, ENDICESOLCLDWVN)
        self.min_drop, self.max_drop, self.min_ice, self.max_ice = min_drop, max_drop, min_ice, max_ice

    def slingo(self, conc_drop, size_drop):
        mask = conc_drop != 0.0
        sd = 0.5 * size_drop
        sd = torch.where(sd < self.min_drop, torch.full_like(sd, self.min_drop),
                         torch.where(sd > self.max_drop, torch.full_like(sd, self.max_drop), sd))
        t = lambda l: torch.as_tensor(l, dtype=conc_drop.dtype, device=conc_drop.device)
        a, b, c, d, e, f = (t(x) for x in (SLINGO_A, SLINGO_B, SLINGO_C, SLINGO_D, SLINGO_E, SLINGO_F))
        sd = sd[..., None]
        ext = 1.0e03 * conc_drop[..., None] * (1.0e-02 * a + (b / sd))
        ssa = 1.0 - (c + d * sd)
        asy = e + 1.0e-03 * f * sd
        return ext, ssa, asy, mask

    def icesolar(self, conc_ice, size_ice):
        mask = conc_ice != 0.0
        si = torch.where(size_ice < self.min_ice, torch.full_like(size_ice, self.min_ice),
                         torch.where(size_ice > self.max_ice, torch.full_like(size_ice, self.max_ice), size_ice))
        a0, a1 = -6.656e-03, 3.686
        B, C, D = _tab(_ICE_B), _tab(_ICE_C), _tab(_ICE_D)
        exts, ssas, asys = [], [], []
        for ni in range(1, 7):
            j = 6 - ni                            # b(7-ni, .) 0-based
            ext = 1.0e03 * conc_ice * (a0 + (a1 / si))
            ssa = 1.0 - (B[0, j] + B[1, j] * si + B[2, j] * si ** 2 + B[3, j] * si ** 3)
            fgam2 = C[0, j] + C[1, j] * si + C[2, j] * si ** 2 + C[3, j] * si ** 3
            fdel2 = D[0, j] + D[1, j] * si + D[2, j] * si ** 2 + D[3, j] * si ** 3
            asy = ((1.0 - fdel2) * fgam2 + 3.0 * fdel2) / 3.0
            exts.append(ext); ssas.append(ssa); asys.append(asy)
        return torch.stack(exts, -1), torch.stack(ssas, -1), torch.stack(asys, -1), mask

    def __call__(self, conc_drop, size_drop, conc_ice, size_ice):
        """cloudpar (nbmax = 1): -> cldext, cldsct, cldasymm (ncol, k, nbands)."""
        sfb = self.spec.solflxbandref
        el, sl, al, ml = self.slingo(conc_drop, size_drop)
        eL, sL, aL = thickavg(*self.liq, sfb, el, sl, al, ml[..., None].expand_as(el)[..., 0])
        ei, si, ai, mi = self.icesolar(conc_ice, size_ice)
        eI, sI, aI = thickavg(*self.ice, sfb, ei, si, ai, mi)
        zero = torch.zeros_like(eL)
        ext = eL + zero + eI + zero
        sct = sL * eL + zero * zero + sI * eI + zero * zero
        asy = (aL * (sL * eL) + zero * (zero * zero) + aI * (sI * eI) + zero * (zero * zero)) / (sct + _tiny(sct))
        return ext, sct, asy


# ---- longwave (cloud_lwpar with lwem_form 'fuliou', lwemiss_calc) --------------------------------
N_EMISS_BDS = 7
ISTARTCLDBAND = [1, 57, 81, 91, 100, 108, 121, 141]
IENDCLDBAND = [56, 80, 90, 99, 107, 120, 140, 220]
ENDFUBANDS = [280, 400, 540, 670, 800, 980, 1100, 1250, 1400, 1700, 1900, 2200]
NBLW = 300
FU_A0 = [-7.752e-03, -1.741e-02, -1.704e-02, -1.151e-02, -1.026e-02, -8.294e-03, -1.153e-02, -9.609e-03,
         -9.061e-03, -8.441e-03, -8.088e-03, -7.770e-03]
FU_A1 = [4.624, 5.541, 4.830, 4.182, 4.105, 3.925, 4.109, 3.768, 3.741, 3.715, 3.717, 3.734]
FU_A2 = [-42.010, -58.420, 16.270, 31.130, 16.360, 1.315, 17.320, 34.110, 26.480, 19.480, 17.170, 11.850]
FU_B = [[0.8079, 0.3964, 0.1028, 0.3254, 0.5207, 0.5631, 0.2307, 0.2037, 0.3105, 0.3908, 0.3014, 0.1996],
        [-0.7004e-02, -0.3155e-02, 0.5019e-02, 0.3434e-02, -0.9778e-03, -0.1434e-02, 0.3830e-02, 0.4247e-02,
         0.2603e-02, 0.1272e-02, 0.2639e-02, 0.3780e-02],
        [0.5209e-04, 0.6417e-04, -0.2024e-04, -0.3081e-04, 0.3725e-05, 0.6298e-05, -0.1616e-04, -0.1810e-04,
         -0.1139e-04, -0.5564e-05, -0.1116e-04, -0.1491e-04],
        [-0.1425e-06, -0.2979e-06, 0.0, 0.9143e-07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]


def fu_lw_weights(lw_band_resolution: float = 10.0):
    """microphys_rad_init: Planck-weighted (233.15 K) contribution of each Fu band to
    each cloud emissivity band -> fulwwts (N_EMISS_BDS, NBFL)."""
    nbfl = len(ENDFUBANDS)
    iendfu = [int((e + 0.01) / lw_band_resolution) for e in ENDFUBANDS]
    xtemv = 233.15
    src1nb = np.zeros(NBLW)
    for n in range(NBLW):
        centnb = 5.0 + n * lw_band_resolution
        c1 = 3.7412e-05 * centnb ** 3
        x = 1.4387 * centnb / xtemv
        src1nb[n] = lw_band_resolution * (c1 / (np.exp(x) - 1.0))
    planck = np.zeros(N_EMISS_BDS)
    for n in range(N_EMISS_BDS):
        for ib in range(ISTARTCLDBAND[n], IENDCLDBAND[n] + 1):
            planck[n] = planck[n] + src1nb[ib - 1]
    for ib in range(ISTARTCLDBAND[N_EMISS_BDS], IENDCLDBAND[N_EMISS_BDS] + 1):
        planck[0] = planck[0] + src1nb[ib - 1]
    nb1 = N_EMISS_BDS + 1
    pivl = np.zeros((nb1, nbfl))
    n1 = np.zeros(nb1, dtype=int); n2 = np.zeros(nb1, dtype=int)
    nivl, s, band = 0, 0.0, 0
    for nw in range(1, NBLW + 1):
        s = s + src1nb[nw - 1]
        if nw == iendfu[nivl]:
            pivl[band, nivl] = s
            s = 0.0
        if nw == IENDCLDBAND[band]:
            if nw != iendfu[nivl]:
                pivl[band, nivl] = s
                s = 0.0
            n2[band] = nivl
            band += 1
            if band < nb1:
                n1[band] = nivl + 1 if nw == iendfu[nivl] else nivl
        if nw == iendfu[nivl]:
            nivl += 1
        if nw >= IENDCLDBAND[N_EMISS_BDS]:
            break
    w = np.zeros((N_EMISS_BDS, nbfl))
    for n in range(N_EMISS_BDS):
        for ni in range(n1[n], n2[n] + 1):
            w[n, ni] = pivl[n, ni] / planck[n]
    for ni in range(n1[N_EMISS_BDS], n2[N_EMISS_BDS] + 1):
        w[0, ni] = pivl[N_EMISS_BDS, ni] / planck[0]
    return w


class CloudLWOptics:
    def __init__(self, min_ice=18.6, max_ice=130.2, alpha=0.1, diffac=1.66, lw_band_resolution=10.0):
        self.w = fu_lw_weights(lw_band_resolution)
        self.min_ice, self.max_ice, self.alpha, self.diffac = min_ice, max_ice, alpha, diffac

    def __call__(self, conc_drop, conc_ice, size_ice, clouddeltaz):
        """-> abscoeff, cldemiss (ncol, k, N_EMISS_BDS)."""
        si = torch.where(size_ice < self.min_ice, torch.full_like(size_ice, self.min_ice),
                         torch.where(size_ice > self.max_ice, torch.full_like(size_ice, self.max_ice), size_ice))
        mask = conc_ice != 0.0
        zero = torch.zeros_like(conc_ice)
        abs_, em = [], []
        for nb in range(N_EMISS_BDS):
            se, ss = zero, zero
            for n in range(len(ENDFUBANDS)):
                ext = 1.0e03 * conc_ice * (FU_A0[n] + FU_A1[n] / si + FU_A2[n] / si ** 2)
                ssa = 1.0 - (FU_B[0][n] + FU_B[1][n] * si + FU_B[2][n] * si ** 2 + FU_B[3][n] * si ** 3)
                se = se + ext * self.w[nb, n]
                ss = ss + ssa * self.w[nb, n]
            ext_i = torch.where(mask, se, zero)
            ssa_i = torch.where(mask, ss, zero)
            a = ext_i * (1.0 - ssa_i) + 1.0e03 * self.alpha * conc_drop + zero * (1.0 - zero) + zero * (1.0 - zero)
            abs_.append(a)
            em.append(1.0 - torch.exp(-self.diffac * a * clouddeltaz * 1.0e-03))
        return torch.stack(abs_, -1), torch.stack(em, -1)
