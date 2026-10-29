"""Lock et al. (2000) boundary-layer entrainment scheme (atmos_param/entrain/entrain.F90).

Surface-driven (convective) and cloud-top radiatively driven mixed layers, their
entrainment velocities and K-profile diffusivities. Translation of `entrain`,
`pbl_depth`, `radml_depth` and `diffusivity_pbl` (siena_201207); diagnostics and the
printing of selected columns are omitted.

The Fortran loops over columns and branches with GOTO; here every column is handled
at once: level searches are masked scans, per-column levels are index tensors, and
the "go to 55" exits become a column mask. kbot is the bottom level (nlev).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..constants import CP_AIR, GRAV, HLS, HLV, RDGAS, RVGAS, TFREEZE, VONKARM
from .colops import add, at, first_true, levels, put
from .monin_obukhov import MoninObukhov
from ..core import spow, ssqrt

SMALL = 1.0e-4
D608 = (RVGAS - RDGAS) / RDGAS


@dataclass(frozen=True)
class EntrainConfig:
    """entrain_nml (defaults of the Fortran; AM2.1 values come from input.nml)."""
    akmax: float = 1.0e4
    wentrmax: float = 0.05
    parcel_buoy: float = 1.0
    frac_inner: float = 0.1
    beta_surf: float = 0.23
    Ashear: float = 25.0
    beta_rad: float = 0.23
    radfmin: float = 30.0
    qdotmin: float = 10.0
    radperturb: float = 0.3
    critjump: float = 0.3
    parcel_option: int = 1
    zcldtopmax: float = 3.0e3
    pr: float = 0.75
    qamin: float = 0.3
    do_jump_exit: bool = True
    apply_entrain: bool = True
    convect_shutoff: bool = False

    @classmethod
    def from_nml(cls, nml: dict) -> "EntrainConfig":
        g = {k.lower(): v for k, v in nml.get("entrain_nml", {}).items()}
        kw = {}
        for f in cls.__dataclass_fields__:
            if f.lower() in g:
                t = cls.__dataclass_fields__[f].type
                kw[f] = {"float": float, "int": int, "bool": bool}[t](g[f.lower()])
        return cls(**kw)


class Entrain:
    def __init__(self, config: EntrainConfig, mo: MoninObukhov):
        self.c = config
        self.mo = mo

    def __call__(self, tdtlw, convect, u_star, b_star, t, qv, ql, qi, qa, u, v,
                 zfull, pfull, zhalf, phalf):
        """Returns k_m_entr, k_t_entr, use_entr (ncol, nlev), zsml, vspblcap (ncol,).

        (The Fortran also overwrites diff_m/diff_t where use_entr = 1; the driver
        recombines with the stable-layer diffusivities, so that output is not needed.)
        """
        c = self.c
        ncol, nlev = t.shape
        ibot = nlev
        zsurf = zhalf[:, nlev]
        zfull_ag = zfull - zsurf[:, None]
        zhalf_ag = zhalf - zsurf[:, None]
        hleff = (torch.clamp(0.05 * (t - TFREEZE + 20.0), 0.0, 1.0) * HLV
                 + torch.clamp(0.05 * (TFREEZE - t), 0.0, 1.0) * HLS)
        if t.dtype == torch.float64:
            slv = CP_AIR * t + GRAV * zfull_ag - hleff * (ql + qi)
            slv = slv * (1 + D608 * (qv + ql + qi))
            svc = slv / CP_AIR
            slvd, tr = slv, 0.0
        else:
            # low precision: svc as an anomaly from tr = t(bottom), so level-to-level and
            # parcel differences of ~300 K values keep their digits; slv = cp (svc + tr)
            tr = t[:, nlev - 1:nlev]
            f = 1 + D608 * (qv + ql + qi)
            svc = (CP_AIR * (t - tr) + GRAV * zfull_ag - hleff * (ql + qi)) * f / CP_AIR + tr * (f - 1.0)
            slvd = CP_AIR * svc
            slv = CP_AIR * (svc + tr)
            tr = tr[:, 0]
        density = pfull / RDGAS / (t * (1.0 + D608 * qv - ql - qi))
        lev = levels(t).expand(ncol, nlev)
        zero = torch.zeros_like(t)
        z1 = torch.zeros_like(u_star)
        k_t_entr = zero
        k_m_entr = zero
        use_entr = zero
        k_t_troen = zero
        k_m_troen = zero
        one_i = torch.ones(ncol, dtype=torch.int64, device=t.device)

        # ---------------- surface-driven convective layer (b_star > 0) ------------------------
        conv = b_star > 0.0
        # Values in columns that do not take a branch must stay finite, or torch.where
        # would pass NaN gradients; the convective branch uses a positive b_star there.
        b_c = torch.where(conv, b_star, torch.full_like(b_star, 1.0e-3))
        # pbl_depth
        svp = svc[:, nlev - 1]
        h1 = zfull_ag[:, nlev - 1]
        ws, _ = self.mo.diff(h1, u_star, b_c)
        ws = torch.clamp(ws / VONKARM / h1, min=SMALL)
        buoy = (c.parcel_buoy * u_star * b_c / GRAV / ws)
        svp = svp * (1.0 + buoy) + tr * buoy       # tr = 0 in float64
        # first k = nlev-1 .. 2 (scanning upward) with svc(k) > svp
        cand = (svc > svp[:, None]) & (lev >= 2) & (lev <= nlev - 1)
        kf, found = first_true(cand, reverse=True)
        kf = torch.where(found, kf, 2 * one_i)
        h2 = at(zfull_ag, kf)
        hb1 = at(zfull_ag, kf + 1)
        t2 = at(svc, kf)
        t1 = at(svc, kf + 1)
        zsml = torch.where(found, h2 + (hb1 - h2) * (t2 - svp) / (t2 - t1), at(zfull_ag, 2 * one_i))
        ipbl = torch.where(found, kf + 1, 2 * one_i)       # not found: k exits the loop at 1
        vsurf3 = u_star * b_c * zsml
        vshear3 = c.Ashear * u_star * u_star * u_star
        # strongest jump: k = ibot .. ipbl+1 (from the bottom), first with jump > critjump
        jump = torch.cat([torch.zeros_like(slv[:, :1]), (slvd[:, :-1] - slvd[:, 1:]) / CP_AIR], dim=1)
        cand = (jump > c.critjump) & (lev >= ipbl[:, None] + 1) & (lev >= 2)
        kj, fj = first_true(cand, reverse=True)
        fj = fj & (ipbl < ibot)
        ipbl = torch.where(fj, kj, ipbl)
        zsml = torch.where(fj, at(zhalf_ag, ipbl), zsml)
        slv_ip = at(slv, ipbl)
        tmp1 = GRAV * torch.clamp((at(slvd, ipbl - 1) - at(slvd, ipbl)) / CP_AIR, min=0.1) / (slv_ip / CP_AIR)
        tmp2 = spow(vsurf3 + vshear3, 2.0 / 3.0) / zsml
        wentr_tmp = torch.clamp(torch.clamp((c.beta_surf * (vsurf3 + vshear3) / zsml) / (tmp1 + tmp2),
                                            min=0.0), max=c.wentrmax)
        k_entr_tmp = torch.clamp(wentr_tmp * (at(zfull_ag, ipbl - 1) - at(zfull_ag, ipbl)), max=c.akmax)
        below = conv[:, None] & (lev >= ipbl[:, None])
        use_entr = torch.where(below, torch.ones_like(use_entr), use_entr)
        k_t_troen = put(k_t_troen, ipbl, k_entr_tmp, conv)
        k_m_troen = put(k_m_troen, ipbl, k_entr_tmp, conv)
        k_t_entr = add(k_t_entr, ipbl, k_entr_tmp, conv)
        k_m_entr = add(k_m_entr, ipbl, k_entr_tmp, conv)
        convpbl = conv
        if c.convect_shutoff:
            shut = conv & convect
            k_t_entr = add(k_t_entr, ipbl, -at(k_t_troen, ipbl), shut)
            k_m_entr = add(k_m_entr, ipbl, -at(k_m_troen, ipbl), shut)
            k_t_troen = put(k_t_troen, ipbl, 0.0, shut)
            k_m_troen = put(k_m_troen, ipbl, 0.0, shut)
        # K profile below ipbl (diffusivity_pbl)
        inner = conv[:, None] & (lev >= ipbl[:, None] + 1) & (ipbl < ibot)[:, None]
        h = zsml
        hinner = c.frac_inner * h
        k_m_ref, k_t_ref = self.mo.diff(hinner, u_star, b_c)
        k_m_mo, k_t_mo = self.mo.diff(zhalf_ag[:, :nlev], u_star[:, None], b_c[:, None])
        zh = zhalf_ag[:, :nlev]
        factor = (zh / hinner[:, None]) * (1.0 - (zh - hinner[:, None]) / (h - hinner)[:, None]) ** 2
        low = zh < hinner[:, None]
        km_p = torch.where(low, k_m_mo, torch.clamp(k_m_ref[:, None] * factor, max=c.akmax))
        kt_p = torch.where(low, k_t_mo, torch.clamp(k_t_ref[:, None] * factor, max=c.akmax))
        k_m_troen = torch.where(inner, km_p, k_m_troen)
        k_t_troen = torch.where(inner, kt_p, k_t_troen)
        k_t_entr = torch.where(inner, k_t_entr + k_t_troen, k_t_entr)
        k_m_entr = torch.where(inner, k_m_entr + k_m_troen, k_m_entr)
        ipbl = torch.where(conv, ipbl, -one_i)

        # ---------------- cloud-top radiatively driven layer ----------------------------------
        cand = zhalf_ag[:, :ibot] < c.zcldtopmax
        kmax, fk = first_true(cand)
        kmax = torch.where(fk, kmax, (ibot + 1) * one_i)
        dpg = (phalf[:, 1:] - phalf[:, :-1]) / GRAV
        radf_all = -1.0 * tdtlw * CP_AIR * dpg
        inrange = lev >= kmax[:, None]
        big_neg = torch.full_like(radf_all, -float("inf"))
        r = torch.where(inrange, radf_all, big_neg)
        rmax, rarg = r.max(dim=1)
        ok1 = rmax > c.radfmin
        kcldtop = torch.where(ok1, rarg + 1, (ibot + 1) * one_i)
        maxradf = torch.where(ok1, rmax, torch.full_like(rmax, c.radfmin))
        qdot = torch.where(inrange, -1.0 * 86400 * tdtlw, big_neg)
        qmax, qarg = qdot.max(dim=1)
        ok2 = ~ok1 & (qmax > c.qdotmin)
        kcldtop = torch.where(ok2, qarg + 1, kcldtop)
        maxradf = torch.where(ok2, at(radf_all, qarg + 1), maxradf)
        rad = ok1 | ok2
        # step up one level if the layer above is less stable, give up if twice
        dec = rad & (at(slvd, kcldtop - 1) < at(slvd, kcldtop))
        kcldtop = torch.where(dec, kcldtop - 1, kcldtop)
        rad = rad & ~(dec & (at(slvd, kcldtop - 1) < at(slvd, kcldtop)))
        kp1 = torch.clamp(kcldtop + 1, max=ibot)
        rad = rad & ~((at(qa, kcldtop - 1) < c.qamin) & (at(qa, kcldtop) < c.qamin) & (at(qa, kp1) < c.qamin))
        # columns without a radiatively driven layer continue with a harmless level
        kcldtop = torch.where(rad, kcldtop, ibot * one_i)
        maxradf = torch.where(rad, maxradf, torch.full_like(maxradf, c.radfmin))
        kp1 = torch.clamp(kcldtop + 1, max=ibot)
        if c.parcel_option == 1:
            svpcp = torch.minimum(at(svc, kcldtop), at(svc, kp1))
        else:
            svpcp = torch.minimum(at(svc, kcldtop), at(svc, kp1) + c.radperturb)
        zradtop = at(zhalf_ag, kcldtop)
        # radml_depth over levels kcldtop..ibot
        svpar = svpcp - c.radperturb
        prev_t = torch.cat([svc[:, :1], svc[:, :-1]], dim=1)          # t1 = svc(k-1)
        prev_h = torch.cat([zfull_ag[:, :1], zfull_ag[:, :-1]], dim=1)
        c1 = svc < svpar[:, None]
        c2 = (prev_t - svc > c.critjump) & (lev > kcldtop[:, None] + 1) if c.do_jump_exit \
            else torch.zeros_like(c1)
        cand = (c1 | c2) & (lev >= kcldtop[:, None] + 1)
        kb, fb = first_true(cand)
        kb = torch.where(fb, kb, kcldtop + 1)
        h2 = at(zfull_ag, kb)
        hh1 = at(prev_h, kb)
        t2 = at(svc, kb)
        tt1 = at(prev_t, kb)
        use_c1 = at(c1.to(t.dtype), kb) > 0.5
        zb = torch.where(use_c1, h2 + (hh1 - h2) * (svpar - t2) / torch.where(use_c1, tt1 - t2, torch.ones_like(t2)),
                         at(zhalf_ag, kb))
        zb = torch.where(fb, zb, z1)
        has_below = kcldtop < ibot
        zradbase = torch.where(has_below, zb, z1)
        zradml = zradtop - zradbase
        rho_top = at(density, kcldtop)
        slv_top = at(slv, kcldtop)
        vrad3 = GRAV * zradml * maxradf / rho_top / slv_top
        vrad = spow(vrad3, 1.0 / 3.0)
        tmp1 = GRAV * torch.clamp(at(svc, kcldtop - 1) - svpcp, min=0.1) / (slv_top / CP_AIR)
        tmp2 = (vrad ** 2.0) / zradml
        wentr_rad = torch.clamp(c.beta_rad * vrad3 / zradml / (tmp1 + tmp2), max=c.wentrmax)
        dzc = at(zfull_ag, kcldtop - 1) - at(zfull_ag, kcldtop)
        k_entr_tmp = torch.clamp(wentr_rad * dzc, max=c.akmax)
        use_entr = put(use_entr, kcldtop, 1.0, rad)
        k_t_entr = add(k_t_entr, kcldtop, k_entr_tmp, rad)
        k_m_entr = add(k_m_entr, kcldtop, k_entr_tmp, rad)
        both = rad & (ipbl == kcldtop) & (ipbl > 0)
        tmp2b = spow(vrad3 + vsurf3 + vshear3, 2.0 / 3.0) / zradml
        wentr_b = torch.clamp(torch.clamp(((c.beta_surf * (vsurf3 + vshear3) + c.beta_rad * vrad3) / zradml)
                                          / (tmp1 + tmp2b), min=0.0), max=c.wentrmax)
        k_b = torch.clamp(wentr_b * dzc, max=c.akmax)
        k_t_troen = put(k_t_troen, ipbl.clamp(min=1), k_b, both)
        k_m_troen = put(k_m_troen, ipbl.clamp(min=1), k_b, both)
        k_t_entr = put(k_t_entr, kcldtop, k_b, both)
        k_m_entr = put(k_m_entr, kcldtop, k_b, both)
        # radiatively driven K profile below cloud top
        ztmp = torch.clamp((zhalf_ag[:, :nlev] - zradbase[:, None]) / zradml[:, None], min=0.0)
        mixed = rad[:, None] & has_below[:, None] & (lev >= kcldtop[:, None] + 1) & (ztmp > 0.0)
        zt_s = torch.where(mixed, ztmp, torch.zeros_like(ztmp))
        k_r = 0.85 * VONKARM * vrad[:, None] * zt_s * zradml[:, None] * zt_s * spow(1.0 - zt_s, 0.5)
        k_r = torch.clamp(k_r, max=c.akmax)
        use_entr = torch.where(mixed, torch.ones_like(use_entr), use_entr)
        k_t_entr = torch.where(mixed, k_t_entr + k_r, k_t_entr)
        k_m_entr = torch.where(mixed, k_m_entr + c.pr * k_r, k_m_entr)
        # cancel surface-driven entrainment below a decoupled cloud layer
        zsml_c = torch.where(conv, zsml, z1)
        cancel = rad & (zradbase < zsml_c) & convpbl & (ipbl > kcldtop)
        ip = ipbl.clamp(min=1)
        k_t_entr = add(k_t_entr, ip, -at(k_t_troen, ip), cancel)
        k_m_entr = add(k_m_entr, ip, -at(k_m_troen, ip), cancel)

        # ---------------- outputs -------------------------------------------------------------
        vspblcap = zhalf_ag[:, 1]
        vspblcap = torch.where(rad, zradbase, vspblcap)
        vspblcap = torch.where(convpbl, z1, vspblcap)
        return dict(k_m_entr=k_m_entr, k_t_entr=k_t_entr, use_entr=use_entr, zsml=zsml_c,
                    vspblcap=vspblcap)
