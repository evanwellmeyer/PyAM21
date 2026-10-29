"""Relaxed Arakawa-Schubert convection (atmos_param/ras/ras.F90, siena_201207).

Moorthi & Suarez (1992) as configured in AM2.1: one cloud type per possible cloud-top
level, called from the lowest to the highest (botop, no random clouds), entraining
plumes rooted at the LCL level of the lowest-model-level parcel, Tokioka entrainment
limit, cloud-work-function relaxation, re-evaporation of precipitation (ras_cevap),
redistribution of the sub-cloud tendencies through the boundary layer (modify_pbl),
and detrainment of cloud liquid/ice/area into the stratiform cloud tracers.
The droplet-number (aerosol activation), online aerosol, tracer transport and cufric
paths are not used by AM2.1 and are not ported.

Vectorisation: the cloud-top level `ic` is the same for every column at a given step
of the cloud loop, while the cloud base (klcl) differs per column. Level loops of the
Fortran therefore run over all levels with a per-column mask `l <= klcl - 1`, and
each Fortran RETURN becomes a column mask. Values in masked-out columns are
discarded with torch.where.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..constants import CP_AIR, GRAV, HLS, HLV, KAPPA
from ..sat_vapor_pres import get_sat_vapor_pres

P00 = 1000.0e2
RHMAX = 0.9999
CP_DIV_GRAV = CP_AIR / GRAV
ONE_PLUS_KAPPA = 1.0 + KAPPA
ONEBCP = 1.0 / CP_AIR
ONEBG = 1.0 / GRAV
CEM = 0.054
CETA = -544.0e-6

_A_DEFAULT = (1.6851, 1.1686, 0.7663, 0.5255, 0.4100, 0.3677, 0.3151, 0.2216,
              0.1521, 0.1082, 0.0750, 0.0664, 0.0553, 0.0445, 0.0633)
_PH_DEFAULT = (150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0,
               550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 850.0)


@dataclass(frozen=True)
class RASConfig:
    fracs: float = 0.25
    rasal0: float = 0.25
    puplim: float = 20.0e2
    aratio: float = 1.4
    cufric: bool = False
    rh_trig: float = 0.0
    alm_min: float = 0.0
    Tokioka_on: bool = False
    Tokioka_con: float = 0.05
    Tokioka_plim: float = 500.0e2
    modify_pbl: bool = False
    prevent_unreasonable: bool = False
    ncrnd: int = 0
    krmin: int = 2
    krmax: int = 2
    botop: bool = True
    rn_ptop: float = 500.0e2
    rn_pbot: float = 800.0e2
    rn_frac_bot: float = 0.8
    rn_frac_top: float = 1.0
    evap_on: bool = True
    cfrac: float = 0.05
    hcevap: float = 0.80
    a: tuple = _A_DEFAULT
    ph: tuple = _PH_DEFAULT

    @classmethod
    def from_nml(cls, nml: dict) -> "RASConfig":
        g = {k.lower(): v for k, v in nml.get("ras_nml", {}).items()}
        kw = {}
        for f in cls.__dataclass_fields__:
            key = f.lower()
            if key not in g:
                continue
            if f in ("a", "ph"):
                vals = list(getattr(cls, f))
                new = g[key] if isinstance(g[key], list) else [g[key]]
                vals[:len(new)] = [float(x) for x in new]
                kw[f] = tuple(vals)
            else:
                kw[f] = type(getattr(cls, f))(g[key])
        return cls(**kw)


class RAS:
    def __init__(self, config: RASConfig = RASConfig()):
        c = config
        if c.ncrnd > 0:
            raise NotImplementedError("random RAS clouds (ncrnd > 0)")
        if c.cufric:
            raise NotImplementedError("RAS cumulus friction (cufric)")
        if c.prevent_unreasonable:
            raise NotImplementedError("RAS prevent_unreasonable")
        self.c = c
        self.rn_pfac = (c.rn_frac_top - c.rn_frac_bot) / (c.rn_pbot - c.rn_ptop)
        # ras_init: critical cloud work function table
        actp, facm = 1.7, 0.01
        self.actop = actp * facm
        a = [facm * x for x in c.a]
        ph = [100.0 * x for x in c.ph]
        tem = [0.0] * 15
        au = [0.0] * 15
        ad = [0.0] * 15
        ac = [0.0] * 15
        for i in range(1, 15):
            tem[i] = ph[i] - ph[i - 1]
            au[i] = a[i - 1] / tem[i]
            ad[i] = a[i] / tem[i]
        for i in range(1, 15):
            ac[i] = ph[i] * au[i] - ph[i - 1] * ad[i]
            ad[i] = ad[i] - au[i]
        self.a, self.ac, self.ad = a, ac, ad

    # ------------------------------------------------------------------------------------------
    def cloud_index(self, kmax: int):
        km1 = kmax - 1
        kcr = min(km1, self.c.krmax)
        kfx = km1 - kcr
        if self.c.botop:
            return [kmax - nc for nc in range(1, kfx + 1)]
        return [kmax - nc for nc in range(kfx, 0, -1)]

    @staticmethod
    def comp_lcl(t_parc, q_parc, p_parc, pres):
        kmax = pres.shape[1]
        svp = get_sat_vapor_pres()
        qsat = svp.compute_qs(t_parc, p_parc, q=q_parc)
        rhum = torch.clamp(q_parc / qsat, max=1.0)
        chi = t_parc / (1669.0 - 122.0 * rhum - t_parc)
        rhum = chi * torch.log(rhum)
        p_lcl = p_parc * torch.exp(rhum)
        p_lcl = torch.maximum(p_lcl, pres[:, 0])
        p_lcl = torch.minimum(p_lcl, p_parc)
        k_lcl = torch.zeros_like(p_lcl, dtype=torch.int64)
        for k in range(2, kmax + 1):
            m = (p_lcl >= pres[:, k - 2]) & (p_lcl <= pres[:, k - 1])
            k_lcl = torch.where(m, torch.full_like(k_lcl, k), k_lcl)
        return torch.clamp(k_lcl, min=kmax // 2)

    # ------------------------------------------------------------------------------------------
    # Level arrays are (ncol, K+2) tensors indexed by the Fortran level l (1..K, 1..K+1 at
    # interfaces; unused slots are 0). Level loops of the Fortran become slices with a
    # per-column mask; running sums and first-order recurrences become cumulative sums,
    # which on the CPU add in the same order as the loops (a masked-out level adds an
    # exact 0), so float64 results are unchanged.

    @staticmethod
    def _g(X, k):
        """X at level k (ncol,) per column."""
        return X.gather(1, k[:, None]).squeeze(1)

    @staticmethod
    def _put(X, lev, k, val):
        return torch.where(lev == k[:, None], val[:, None], X)

    @staticmethod
    def _setcol(X, l, val):
        return torch.cat([X[:, :l], val[:, None], X[:, l + 1:]], dim=1)

    @staticmethod
    def _setslice(X, lo, hi, val):
        return torch.cat([X[:, :lo], val, X[:, hi:]], dim=1)

    @staticmethod
    def _rcum(T, init=None):
        """Running sums over the levels of T taken from the last to the first (the Fortran's
        downward-counting loops): R[:, j] = init + T[:, -1] + ... + T[:, j]. Returns R and
        the values before each addition (B[:, j] = R[:, j+1], B[:, -1] = init)."""
        seq = T.flip(1)
        if init is not None:
            seq = torch.cat([init[:, None], seq], dim=1)
        cs = seq.cumsum(1)
        if init is None:
            R = cs.flip(1)
            B = torch.cat([cs[:, :1] * 0.0, cs[:, :-1]], dim=1).flip(1)
        else:
            R = cs[:, 1:].flip(1)
            B = cs[:, :-1].flip(1)
        return R, B

    def cloud_exist(self, k, qvap, qsat, theta, pi_int, ic, Hl):
        """ras_cloud_exist for cloud tops down to level ic (min of the cloud index list)."""
        g = self._g
        K = qvap.shape[1] - 2
        qol_k = torch.minimum(g(qsat, k) * RHMAX, g(qvap, k))
        ssl_k = g(pi_int, k + 1) * g(theta, k) * CP_AIR
        hol_k = ssl_k + qol_k * Hl
        hst_k = ssl_k + g(qsat, k) * Hl
        zzl0 = (g(pi_int, k + 1) - g(pi_int, k)) * g(theta, k) * CP_AIR
        lev = torch.arange(ic + 1, K, device=k.device)
        m = lev[None, :] <= (k - 1)[:, None]
        th, qs = theta[:, ic + 1:K], qsat[:, ic + 1:K]
        term = torch.where(m, (pi_int[:, ic + 2:K + 1] - pi_int[:, ic + 1:K]) * th * CP_AIR, torch.zeros_like(th))
        _, zb = self._rcum(term, zzl0)
        ssl = zb + pi_int[:, ic + 2:K + 1] * th * CP_AIR
        hst = ssl + qs * Hl[:, None]
        inf = torch.full_like(hst, float("inf"))
        hmin = torch.minimum(hst_k, torch.where(m, hst, inf).amin(dim=1)) if hst.shape[1] else hst_k
        zzl = zb[:, 0] + term[:, 0] if hst.shape[1] else zzl0
        ssl = zzl + pi_int[:, ic + 1] * theta[:, ic] * CP_AIR
        hmin = torch.minimum(hmin, ssl + qsat[:, ic] * Hl)
        return hol_k > hmin

    # ------------------------------------------------------------------------------------------
    def ras_cloud(self, k, ic, active, rasal, frac, Hl, coldT, theta, qvap, pres_int, pi_int, pi,
                  psfc, alf, bet, gam, cp_by_dp, zbase, ql, qi, qa):
        """One cloud type (top at level ic, base at k) for all columns. Level arrays are
        (ncol, K+2). Returns dtcu, dqcu, mccu, Dl, Di, Da (ncol, K+2) and dpcu, almx (ncol,),
        all zero where the cloud does not form."""
        c = self.c
        g, put, setcol, setslice = self._g, self._put, self._setcol, self._setslice
        C, K = theta.shape[0], theta.shape[1] - 2
        dev = theta.device
        lev = torch.arange(K + 2, device=dev)[None, :]
        z = torch.zeros_like(psfc)
        Z = torch.zeros_like(theta)
        km1 = k - 1
        ic1 = ic + 1
        HlC = Hl[:, None]
        # levels ic+1 .. K-1 (the Fortran's l = km1 .. ic1, masked per column)
        a, b = ic + 1, K
        levL = torch.arange(a, b, device=dev)[None, :]
        m = levL <= km1[:, None]
        zL = torch.zeros(C, b - a, dtype=theta.dtype, device=dev)

        th_k = g(theta, k)
        qs1 = g(alf, k) + g(bet, k) * th_k
        qol_k = torch.minimum(qs1 * RHMAX, g(qvap, k))
        pik1 = g(pi_int, k + 1)
        hol_k = pik1 * th_k * CP_AIR + qol_k * Hl
        zzl0 = (pik1 - g(pi_int, k)) * th_k * CP_AIR
        thL = theta[:, a:b]
        qs1l = alf[:, a:b] + bet[:, a:b] * thL
        qolL = torch.where(m, torch.minimum(qs1l * RHMAX, qvap[:, a:b]), zL)
        dtempL = (pi_int[:, a + 1:b + 1] - pi_int[:, a:b]) * thL
        zzlR, zzlB = self._rcum(torch.where(m, dtempL * CP_AIR, zL), zzl0)
        zzl = zzlR[:, 0] if b > a else zzl0
        ssl = zzlB + pi_int[:, a + 1:b + 1] * thL * CP_AIR
        holL = torch.where(m, ssl + qolL * HlC, zL)
        hstL = torch.where(m, ssl + qs1l * HlC, zL)
        etaL, _ = self._rcum(torch.where(m, dtempL * CP_DIV_GRAV, zL))
        qol = put(setslice(Z, a, b, qolL), lev, k, qol_k)
        hol = put(setslice(Z, a, b, holL), lev, k, hol_k)
        hst = setslice(Z, a, b, hstL)
        eta = setslice(Z, a, b, etaL)
        qs1 = alf[:, ic] + bet[:, ic] * theta[:, ic]
        qol_ic = torch.minimum(qs1 * RHMAX, qvap[:, ic])
        ssl = zzl + pi_int[:, ic1] * theta[:, ic] * CP_AIR
        hol_ic = ssl + qol_ic * Hl
        hst_ic = ssl + qs1 * Hl
        dtemp = (pi_int[:, ic1] - pi[:, ic]) * theta[:, ic]
        eta_ic = eta[:, ic1] + dtemp * CP_DIV_GRAV
        qol, hol, hst, eta = setcol(qol, ic, qol_ic), setcol(hol, ic, hol_ic), setcol(hst, ic, hst_ic), \
            setcol(eta, ic, eta_ic)

        # --- cloud work function and entrainment (levels ic .. K-1)
        levI = torch.arange(ic, K, device=dev)[None, :]
        mI = levI <= km1[:, None]
        hol_k = g(hol, k)
        xx1 = hol_k - hst_ic
        tI = torch.where(mI, (hst_ic[:, None] - hol[:, ic:K]) * (eta[:, ic:K] - eta[:, ic + 1:K + 1]),
                         torch.zeros_like(eta[:, ic:K]))
        xx2 = tI.cumsum(1)[:, -1] if K > ic else z
        lcase1 = (xx2 > 0.0) & (xx1 > 0.0)
        lcase2 = (xx1 <= 0.0) & (hol_k > hst[:, ic1]) & (ic1 < k)
        ok = active & (lcase1 | lcase2)
        alm = torch.where(lcase1, xx1 / torch.where(lcase1, xx2, torch.ones_like(xx2)), z)
        xlim = torch.full_like(z, c.alm_min)
        if c.Tokioka_on:
            xp = 0.5 * (pres_int[:, ic] + pres_int[:, ic1])
            tok = xp <= c.Tokioka_plim
            xlim = torch.where(tok, c.Tokioka_con / torch.where(tok, zbase, torch.ones_like(zbase)), xlim)
        ok = ok & ~(alm < xlim)
        eta = setslice(eta, ic, K, torch.where(mI, 1.0 + alm[:, None] * eta[:, ic:K], eta[:, ic:K]))
        eta = put(eta, lev, k, torch.ones_like(z))
        eta_ic = eta[:, ic]
        # wfn: l = K-1 .. ic+1
        inc = torch.where(m, (eta[:, a:b] - eta[:, a + 1:b + 1]) * hol[:, a:b], zL)
        hccR, hccB = self._rcum(inc, hol_k)
        hcc = hccB + (eta[:, a:b] - eta[:, a + 1:b + 1]) * hol[:, a:b]
        dpib = pi_int[:, a + 1:b + 1] - pi[:, a:b]
        dpit = pi[:, a:b] - pi_int[:, a:b]
        a1 = (eta[:, a + 1:b + 1] * dpib + eta[:, a:b] * dpit) * hst[:, a:b]
        a2 = hccB * dpib + hcc * dpit
        wR, _ = self._rcum(torch.where(m, (a2 - a1) * gam[:, a:b], zL))
        wfn = wR[:, 0] if b > a else z
        hccp = hccR[:, 0] if b > a else hol_k
        wfn = torch.where(lcase1, wfn + gam[:, ic] * (pi_int[:, ic1] - pi[:, ic]) * (hccp - hst_ic * eta[:, ic1]),
                          wfn)
        xx1 = 0.5 * (pres_int[:, ic] + pres_int[:, ic1])
        xx2 = g(pres_int, k)
        ftop = torch.ones_like(z)
        dh = hst[:, ic1] - hst_ic
        ftop2 = torch.where(hst[:, ic1] < hst_ic, (hst[:, ic1] - hol_k) / torch.where(dh != 0, dh, torch.ones_like(dh)),
                            z)
        ftop = torch.where(lcase2, ftop2, ftop)
        if ic1 + 1 <= K + 1:
            xx3 = 0.5 * (pres_int[:, ic1] + pres_int[:, ic1 + 1])
            xx1 = torch.where(lcase2, xx3 * (1.0 - ftop) + xx1 * ftop, xx1)
        iwk = torch.trunc(xx1 * 0.02e-2 - 0.999999999).to(torch.int64)
        ac_t = torch.tensor([0.0] + self.ac, dtype=z.dtype, device=dev)
        ad_t = torch.tensor([0.0] + self.ad, dtype=z.dtype, device=dev)
        ii = iwk.clamp(1, 15)
        wfn_crit = torch.where((iwk > 1) & (iwk <= 15), ac_t[ii] + xx1 * ad_t[ii],
                               torch.where(iwk <= 1, torch.full_like(z, self.actop),
                                           torch.full_like(z, self.a[14])))
        wfn_crit = c.aratio * wfn_crit * (xx2 - xx1)
        wfn = wfn - wfn_crit
        lcase1 = lcase1 & (wfn > 0.0)
        lcase2 = lcase2 & (wfn > 0.0) & (ftop > 0.0)
        ok = ok & (lcase1 | lcase2)
        almx = torch.where(ok, alm, z)
        # --- cloud properties
        dhic = torch.where(lcase1, hst_ic - hol_ic, (hol_k - hol[:, ic1]) - (hol_ic - hol[:, ic1]) * ftop)
        xa = ftop * (hol_ic - hol[:, ic1]) + hol[:, ic1]
        xb = ftop * (qol_ic - qol[:, ic1]) + qol[:, ic1]
        sic = torch.where(lcase2, xa - xb * Hl, z)
        qs1 = torch.where(lcase2, xb + dhic * (1.0 / Hl), qs1)
        hic = hol_ic
        hkb = hol_k
        # levels ic .. K-1 (l = km1 .. ic)
        dE = eta[:, ic:K] - eta[:, ic + 1:K + 1]
        zI = torch.zeros_like(dE)
        wlqR, _ = self._rcum(torch.where(mI, dE * qol[:, ic:K], zI), g(qol, k) - qs1 * eta_ic)
        wllR, _ = self._rcum(torch.where(mI, dE * ql[:, ic:K], zI), g(ql, k))
        wliR, _ = self._rcum(torch.where(mI, dE * qi[:, ic:K], zI), g(qi, k))
        wlq, wll, wli = wlqR[:, 0], wllR[:, 0], wliR[:, 0]
        # --- detrained cloud water, ice and area and mass flux
        cpk = g(cp_by_dp, k)
        xk = 0.5 * cpk / CP_AIR / ONEBG
        qlm, qim, qam = g(ql, km1), g(qi, km1), g(qa, km1)
        qlk, qik, qak = g(ql, k), g(qi, k), g(qa, k)
        xL = 0.5 * cp_by_dp[:, a:b] / CP_AIR / ONEBG
        eL, eL1 = eta[:, a:b], eta[:, a + 1:b + 1]

        def det(q, qm, qk):
            v = (eL1 * (q[:, a:b] - q[:, a + 1:b + 1]) + eL * (q[:, a - 1:b - 1] - q[:, a:b])) * xL
            return put(setslice(Z, a, b, torch.where(m, v, zL)), lev, k, (qm - qk) * xk)

        Dl, Di, Da = det(ql, qlm, qlk), det(qi, qim, qik), det(qa, qam, qak)
        mc = put(setslice(Z, a, b, torch.where(m, eL, zL)), lev, k, g(eta, k))
        x1 = cp_by_dp[:, ic] / CP_AIR / ONEBG
        x2 = 0.5 * x1
        Dl_ic = (eta[:, ic1] * (ql[:, ic] - ql[:, ic1]) * x2) + (wll - eta_ic * ql[:, ic]) * x1
        Di_ic = (eta[:, ic1] * (qi[:, ic] - qi[:, ic1]) * x2) + (wli - eta_ic * qi[:, ic]) * x1
        Da_ic = (eta[:, ic1] * (qa[:, ic] - qa[:, ic1]) * x2) + (eta_ic - eta_ic * qa[:, ic]) * x1
        # --- heating and moistening profiles (hol and gmh are reused)
        thm, thk = g(theta, km1), g(theta, k)
        pim, pik = g(pi, km1), g(pi, k)
        piik = g(pi_int, k)
        cpm = g(cp_by_dp, km1)
        x = (thm - thk) / (pik - pim)
        hk_new = x * (piik - pim) * pik * cpk
        hm_new = x * (pik - piik) * pim * cpm
        holP = put(put(hol, lev, k, hk_new), lev, km1, hm_new)
        # step l (active): hol(l) += X (pint(l)-pi(l-1)) pi(l) cp(l); hol(l-1) = X (pi(l)-pint(l)) pi(l-1) cp(l-1)
        X = (theta[:, a - 1:b - 1] - thL) * eL / (pi[:, a:b] - pi[:, a - 1:b - 1])
        Aadd = X * (pi_int[:, a:b] - pi[:, a - 1:b - 1]) * pi[:, a:b] * cp_by_dp[:, a:b]
        Bv = X * (pi[:, a:b] - pi_int[:, a:b]) * pi[:, a - 1:b - 1] * cp_by_dp[:, a - 1:b - 1]   # -> hol(l-1)
        Bnext = torch.cat([Bv[:, 1:], zL[:, :1]], dim=1)                                            # Bv at l+1
        prev = torch.where(levL == km1[:, None], holP[:, a:b], Bnext)
        holL = torch.where(m, Aadd + prev, holP[:, a:b])
        hol_ic = torch.where(km1 >= ic + 1, Bv[:, 0], holP[:, ic]) if b > a else holP[:, ic]
        hol = setcol(setslice(holP, a, b, holL), ic, hol_ic)
        akR, _ = self._rcum(torch.where(m, -(holL * (eL * (pi[:, a:b] - pi_int[:, a:b])
                                                     + eL1 * (pi_int[:, a + 1:b + 1] - pi[:, a:b])) / pi[:, a:b]), zL))
        akm = akR[:, 0] if b > a else z
        xp = 0.5 * (pres_int[:, ic] + pres_int[:, ic1])
        rn_frac = torch.full_like(z, c.rn_frac_top)
        rn_frac = torch.where((xp >= c.rn_ptop) & (xp <= c.rn_pbot), (c.rn_pbot - xp) * self.rn_pfac + c.rn_frac_bot,
                              rn_frac)
        rn_frac = torch.where(xp > c.rn_pbot, torch.full_like(z, c.rn_frac_bot), rn_frac)
        wdet = (1.0 - rn_frac) * wlq
        wlq = rn_frac * wlq
        det_add = wdet * cp_by_dp[:, ic] / CP_AIR / ONEBG
        Di_ic = torch.where(coldT, Di_ic + det_add, Di_ic)
        Dl_ic = torch.where(coldT, Dl_ic, Dl_ic + det_add)
        wdet = z
        xh = hol_ic
        xh = torch.where(lcase2, xh + (sic - hic + qol_ic * Hl) * (cp_by_dp[:, ic] / CP_AIR), xh)
        hol_ic = xh - (wdet * Hl * cp_by_dp[:, ic] / CP_AIR)
        akm = torch.where(lcase1, akm - eta[:, ic1] * (pi_int[:, ic1] - pi[:, ic]) * xh / pi[:, ic], akm)
        xq0 = g(qol, km1) - g(qol, k)
        gk = g(hol, k) + (xq0 * cpk * Hl * 0.5 / CP_AIR)
        akm = akm + g(gam, km1) * (piik - pim) * gk
        # gmh: l = K-1 .. ic+1
        xqn = (qol[:, a - 1:b - 1] - qol[:, a:b]) * eL
        xqprev = torch.where(levL == km1[:, None], xq0[:, None], torch.cat([xqn[:, 1:], zL[:, :1]], dim=1))
        xx3 = xqprev + xqn
        gmhL = torch.where(m, holL + (xx3 * cp_by_dp[:, a:b] * HlC * 0.5 / CP_AIR), zL)
        xq = torch.where(km1 >= ic + 1, xqn[:, 0], xq0) if b > a else xq0
        gmh = put(setslice(Z, a, b, gmhL), lev, k, gk)
        xq = torch.where(lcase2, xq + 2.0 * (hkb - dhic - sic - qol_ic * Hl) / Hl, xq)
        gmh_ic = xh + cp_by_dp[:, ic] * ONEBCP * (xq * Hl * 0.5 + eta_ic * dhic)
        # xg / akm: l = K-1 .. ic+1
        _, xgB = self._rcum(torch.where(m, (eL - eL1) * gmhL, zL), g(gmh, k))
        xgn = xgB + (eL - eL1) * gmhL
        y2 = gam[:, a - 1:b - 1] * (pi_int[:, a:b] - pi[:, a - 1:b - 1])
        if b > a:
            y2 = torch.cat([torch.where(lcase2, z, y2[:, 0])[:, None], y2[:, 1:]], dim=1)
        akR, _ = self._rcum(torch.where(m, xgn * (y2 + gam[:, a:b] * (pi[:, a:b] - pi_int[:, a:b])), zL), akm)
        akm = akR[:, 0] if b > a else akm
        hol = setcol(hol, ic, hol_ic)
        gmh = setcol(gmh, ic, gmh_ic)
        if ic + 2 <= K + 1:
            y1 = 0.5 * (pres_int[:, ic] + pres_int[:, ic1]) + 0.5 * (pres_int[:, ic + 2] - pres_int[:, ic]) * (1.0 - ftop)
            y2s = pres_int[:, ic1]
            y3 = 0.5 * (pres_int[:, ic1] + pres_int[:, ic + 2])
            inb = lcase2 & (y1 >= y2s) & (y1 < y3)
            ftop = torch.where(inb, 1.0 - (y1 - y2s) / (y3 - y2s), ftop)
            ftop = torch.where(lcase2 & (y1 < y2s), torch.ones_like(z), ftop)
            ftop = torch.where(lcase2 & ~(y1 < y3), z, ftop)
            xx4 = cp_by_dp[:, ic1] / cp_by_dp[:, ic]
            hol = setcol(hol, ic1, torch.where(inb, hol[:, ic1] + hol[:, ic] * xx4, hol[:, ic1]))
            gmh = setcol(gmh, ic1, torch.where(inb, gmh[:, ic1] + gmh[:, ic] * xx4, gmh[:, ic1]))
            hol = setcol(hol, ic, torch.where(inb, z, hol[:, ic]))
            gmh = setcol(gmh, ic, torch.where(inb, z, gmh[:, ic]))
        Dl, Di, Da = setcol(Dl, ic, Dl_ic), setcol(Di, ic, Di_ic), setcol(Da, ic, Da_ic)
        # --- cloud-base mass flux
        rel = (akm < 0.0) & (wlq >= 0.0)
        rasalf = rasal * (pres_int[:, ic + 1] - c.puplim) / (psfc - c.puplim)
        rasalf = torch.where(c.puplim > psfc, z, rasalf)
        rasalf = torch.clamp(rasalf, min=0.0)
        akm_s = torch.where(rel, akm, -torch.ones_like(akm))
        wfn = torch.where(rel, -ftop * wfn * rasalf / akm_s, z)
        wfn = torch.minimum(wfn, (psfc - g(pres_int, k)) * frac)
        wfn = torch.where(ok, wfn, z)
        x = (wfn * ONEBG)[:, None]
        xt = (wfn * ONEBCP)[:, None]
        xqq = (wfn * (1.0 / Hl))[:, None]
        mO = ok[:, None] & (lev >= ic) & (lev <= k[:, None]) & (lev <= K)
        piS = torch.where((lev >= 1) & (lev <= K), pi, torch.ones_like(pi))
        Da = torch.where(mO, Da * x, Z)
        Dl = torch.where(mO, Dl * x, Z)
        Di = torch.where(mO, Di * x, Z)
        mc = torch.where(mO, mc * x, Z)
        dtcu = torch.where(mO, xt * hol / piS, Z)
        dqcu = torch.where(mO, xqq * (gmh - hol), Z)
        dpcu = torch.where(ok, wlq * wfn * ONEBG, z)
        return dict(dtcu=dtcu, dqcu=dqcu, mccu=mc, Dl=Dl, Di=Di, Da=Da, dpcu=dpcu, almx=almx)

    # ------------------------------------------------------------------------------------------
    def cevap(self, ib, temp, qvap, pres, mass, qvap_sat, dqvap_sat, psfc, Hl, dtime, dpcu):
        """ras_cevap for cloud-top level ib; returns dtevap, dqevap (ncol, K+2) and dpevap."""
        c = self.c
        K = temp.shape[1] - 2
        z = torch.zeros_like(psfc)
        dpevap = z
        dte = [z] * (K + 2)
        dqe = [z] * (K + 2)
        for k in range(ib + 1, K + 1):
            prec = torch.clamp(dpcu - dpevap, min=0.0)
            prec_mmph = prec * 3600.0 / dtime
            pfac = torch.sqrt(pres[:, k] / psfc)
            arg = CEM * c.cfrac * prec_mmph * pfac
            pos = arg > 0.0
            emx = torch.where(pos, torch.sqrt(torch.where(pos, arg, torch.ones_like(arg))), z)
            evef = 1.0 - torch.exp(CETA * dtime * emx)
            cond = (c.hcevap * qvap_sat[:, k] >= qvap[:, k]) & (prec > 0.0)     # ksfc > k always
            d = (c.hcevap * qvap_sat[:, k] - qvap[:, k]) / (1.0 + (Hl * c.hcevap * dqvap_sat[:, k] / CP_AIR))
            d = evef * d
            d = torch.minimum(d, prec / mass[:, k])
            qn = qvap[:, k] + d
            tn = temp[:, k] - (d * Hl / CP_AIR)
            dqe[k] = torch.where(cond, qn - qvap[:, k], z)
            dte[k] = torch.where(cond, tn - temp[:, k], z)
            dpevap = torch.where(cond, dpevap + d * mass[:, k], dpevap)
        return torch.stack(dte, 1), torch.stack(dqe, 1), dpevap

    # ------------------------------------------------------------------------------------------
    def __call__(self, dtime, temp0, qvap0, uwnd0, vwnd0, pres0, pres0_int, zhalf0, coldT0, ql0, qi0, qa0):
        """RAS with the stratiform cloud tracers. Arrays (ncol, nlev) [(ncol, nlev+1) at
        interfaces]; coldT0 (ncol,) bool.

        Returns the updated state (t, q, u, v, ql, qi, qa), the tendencies divided by
        dtime (ttnd, qtnd, utnd, vtnd, qltnd, qitnd, qatnd), rain, snow (kg/m2/s),
        mc (ncol, nlev+1) and det0 (ncol, nlev).
        """
        c = self.c
        C, K = temp0.shape
        dev = temp0.device
        g = self._g
        svp = get_sat_vapor_pres()
        qvap_sat0, dqvap_sat0 = svp.compute_qs(temp0, pres0, want_dqsdT=True)
        q_parc = torch.clamp(qvap0[:, K - 1], min=1.0e-6)
        q_parc = torch.minimum(q_parc, qvap_sat0[:, K - 1])
        klcl = self.comp_lcl(temp0[:, K - 1], q_parc, pres0[:, K - 1], pres0)
        rhtrig = q_parc >= c.rh_trig * qvap_sat0[:, K - 1]
        psfc = pres0_int[:, K]
        frac = c.fracs / dtime
        rasal = c.rasal0 / dtime
        Hl = torch.where(coldT0, torch.full_like(psfc, HLS), torch.full_like(psfc, HLV))
        dp = torch.clamp(pres0_int[:, 1:] - pres0_int[:, :-1], min=1.0e-5)
        zbase = zhalf0.gather(1, (klcl - 1)[:, None]).squeeze(1) - zhalf0[:, K]
        pi_int_t = (pres0_int / P00) ** KAPPA
        pi_t = (pi_int_t[:, 1:] * pres0_int[:, 1:] - pi_int_t[:, :-1] * pres0_int[:, :-1]) / (dp * ONE_PLUS_KAPPA)
        pi_t = torch.clamp(pi_t, min=1.0e-5)
        theta_t = temp0 / pi_t
        cp_by_dp_t = CP_AIR / dp
        mass_t = dp / GRAV

        z1 = torch.zeros(C, 1, dtype=temp0.dtype, device=dev)
        Pd = lambda x: torch.cat([z1, x, z1], dim=1)             # full levels -> (C, K+2)
        pres_int, pi_int = torch.cat([z1, pres0_int], 1), torch.cat([z1, pi_int_t], 1)
        pi, cp_by_dp, mass, pres = Pd(pi_t), Pd(cp_by_dp_t), Pd(mass_t), Pd(pres0)
        qvap_sat, dqvap_sat = Pd(qvap_sat0), Pd(dqvap_sat0)
        temp, theta, qvap = Pd(temp0), Pd(theta_t), Pd(qvap0)
        ql, qi, qa = Pd(ql0), Pd(qi0), Pd(qa0)
        # alpha, beta, gamma are set at the first cloud from the initial column (setras)
        alf = qvap_sat - dqvap_sat * temp
        bet = dqvap_sat * pi
        gam = 1.0 / ((1.0 + (Hl[:, None] * dqvap_sat / CP_AIR)) * torch.where(pi != 0, pi, torch.ones_like(pi)))

        ic_list = self.cloud_index(K)
        exist = rhtrig & self.cloud_exist(klcl, qvap, qvap_sat, theta, pi_int, min(ic_list), Hl)
        z = torch.zeros_like(psfc)
        Zl = torch.zeros_like(temp)
        lev = torch.arange(K + 2, device=dev)[None, :]
        inner = (lev >= 1) & (lev <= K)
        dtemp = dqv = Dl_acc = Di_acc = Da_acc = det = Zl
        mc = Zl
        piS = torch.where(inner, pi, torch.ones_like(pi))
        precip = z
        for ib in ic_list:
            act = exist & (ib < klcl)
            # a cloud type active in no column changes nothing (every update is masked)
            if not bool(act.any()):
                continue
            r = self.ras_cloud(klcl, ib, act, rasal, frac, Hl, coldT0, theta, qvap, pres_int, pi_int, pi,
                               psfc, alf, bet, gam, cp_by_dp, zbase, ql, qi, qa)
            dtcu = r["dtcu"] * dtime
            dqcu = r["dqcu"] * dtime
            dpcu = r["dpcu"] * dtime
            Da, Dl, Di = r["Da"] * dtime, r["Dl"] * dtime, r["Di"] * dtime
            mccu = r["mccu"]
            A = act[:, None]
            if c.modify_pbl:
                mc_lcl = g(mccu, klcl)
                pk = g(pres_int, klcl)
                mpbl = A & (lev >= 2) & (lev >= klcl[:, None] + 1)
                mccu = torch.where(mpbl, (mc_lcl[:, None] * (psfc[:, None] - pres_int) / (psfc - pk)[:, None]),
                                   mccu)
                dpfac = (g(pres_int, klcl + 1) - pk) / (psfc - pk)
                dtcu_pbl = dpfac * g(dtcu, klcl) * g(pi, klcl)
                dqcu_pbl = dpfac * g(dqcu, klcl)
                mp = A & inner & (lev >= klcl[:, None])
                dtcu = torch.where(mp, dtcu_pbl[:, None] / piS, dtcu)
                dqcu = torch.where(mp, dqcu_pbl[:, None], dqcu)
            ev = act & (dpcu > 0.0)
            if c.evap_on and bool(ev.any()):
                dte, dqe, dpevap = self.cevap(ib, temp, qvap, pres, mass, qvap_sat, dqvap_sat, psfc, Hl,
                                              dtime, dpcu)
                E = ev[:, None] & inner
                dtcu = torch.where(E, dtcu + dte / piS, dtcu)
                dqcu = torch.where(E, dqcu + dqe, dqcu)
                dpcu = torch.where(ev, torch.clamp(dpcu - dpevap, min=0.0), dpcu)
            Ai = A & inner
            ql = torch.where(Ai, ql + Dl, ql)
            qi = torch.where(Ai, qi + Di, qi)
            qa = torch.where(Ai, qa + Da, qa)
            theta = torch.where(Ai, theta + dtcu, theta)
            qvap = torch.where(Ai, qvap + dqcu, qvap)
            temp = torch.where(Ai, theta * pi, temp)
            dtemp = torch.where(Ai, dtemp + dtcu * pi, dtemp)
            dqv = torch.where(Ai, dqv + dqcu, dqv)
            Da_acc = torch.where(Ai, Da_acc + Da, Da_acc)
            Dl_acc = torch.where(Ai, Dl_acc + Dl, Dl_acc)
            Di_acc = torch.where(Ai, Di_acc + Di, Di_acc)
            mc = torch.where(A & (lev >= 1), mc + mccu, mc)
            det = torch.where(A & (lev == ib), det + mccu[:, ib + 1:ib + 2], det)
            precip = torch.where(act, precip + dpcu, precip)

        S = lambda X: X[:, 1:K + 1]
        dtemp0 = S(dtemp)
        dqvap0 = S(dqv)
        Dl0, Di0, Da0 = S(Dl_acc), S(Di_acc), S(Da_acc)
        dtinv = 1.0 / dtime
        rain = torch.where(coldT0, z, precip) * dtinv
        snow = torch.where(coldT0, precip, z) * dtinv
        zero3 = torch.zeros_like(temp0)
        return dict(t=temp0 + dtemp0, q=qvap0 + dqvap0, u=uwnd0 + zero3, v=vwnd0 + zero3,
                    ql=ql0 + Dl0, qi=qi0 + Di0, qa=qa0 + Da0,
                    ttnd=dtemp0 * dtinv, qtnd=dqvap0 * dtinv, utnd=zero3 * dtinv, vtnd=zero3 * dtinv,
                    qltnd=Dl0 * dtinv, qitnd=Di0 * dtinv, qatnd=Da0 * dtinv,
                    rain=rain, snow=snow, mc=mc[:, 1:K + 2], det0=S(det))
