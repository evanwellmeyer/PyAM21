"""Monin-Obukhov similarity theory (atmos_param/monin_obukhov, siena_201207).

Translation of monin_obukhov_kernel.F90 with the namelist handling of
monin_obukhov.F90: surface drag coefficients by Newton iteration on zeta
(mo_drag), profiles between the surface and a reference height (mo_profile),
surface-layer diffusivities (mo_diff) and the stable mixing function (stable_mix).

All functions are elementwise over tensors of any shape. The Newton iteration always
runs max_iter sweeps with per-point masks; a point stops changing when its own
correction is below the tolerance, which is what the Fortran's vector loop does (it
exits only when every point has converged, and converged points are masked out).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..constants import GRAV, VONKARM


@dataclass(frozen=True)
class MoninObukhovConfig:
    """monin_obukhov_nml (AM2.1: stable_option = 2, rich_crit = 10, zeta_trans = 0.5)."""
    rich_crit: float = 2.0
    drag_min: float = 1.0e-05
    neutral: bool = False
    stable_option: int = 1
    zeta_trans: float = 0.5

    @classmethod
    def from_nml(cls, nml: dict) -> "MoninObukhovConfig":
        g = nml.get("monin_obukhov_nml", {})
        return cls(rich_crit=float(g.get("rich_crit", 2.0)), drag_min=float(g.get("drag_min", 1.0e-5)),
                   neutral=bool(g.get("neutral", False)), stable_option=int(g.get("stable_option", 1)),
                   zeta_trans=float(g.get("zeta_trans", 0.5)))


class MoninObukhov:
    ERROR = 1.0e-04
    ZETA_MIN = 1.0e-06
    MAX_ITER = 20
    SMALL = 1.0e-04
    USTAR_MIN = 1.0e-10

    def __init__(self, config: MoninObukhovConfig = MoninObukhovConfig()):
        self.c = config
        self.b_stab = 1.0 / config.rich_crit
        self.lam = 1.0 + (5.0 - self.b_stab) * config.zeta_trans
        self.rich_trans = config.zeta_trans / (1.0 + 5.0 * config.zeta_trans)

    # ---- similarity functions ---------------------------------------------------------------
    # Each branch is evaluated on zeta clamped to its own sign range, so the branch not
    # taken stays finite and torch.where gives finite gradients.

    def derivative_m(self, zeta):
        c = self.c
        unstable = zeta < 0.0
        zu, zs = torch.clamp(zeta, max=0.0), torch.clamp(zeta, min=0.0)
        x = (1 - 16.0 * zu) ** (-0.5)
        phi_u = torch.sqrt(x)
        if c.stable_option == 1:
            phi_s = 1.0 + zs * (5.0 + self.b_stab * zs) / (1.0 + zs)
        else:
            phi_s = torch.where(zs < c.zeta_trans, 1 + 5.0 * zs, self.lam + self.b_stab * zs)
        return torch.where(unstable, phi_u, phi_s)

    def derivative_t(self, zeta):
        c = self.c
        unstable = zeta < 0.0
        zu, zs = torch.clamp(zeta, max=0.0), torch.clamp(zeta, min=0.0)
        phi_u = (1 - 16.0 * zu) ** (-0.5)
        if c.stable_option == 1:
            phi_s = 1.0 + zs * (5.0 + self.b_stab * zs) / (1.0 + zs)
        else:
            phi_s = torch.where(zs < c.zeta_trans, 1 + 5.0 * zs, self.lam + self.b_stab * zs)
        return torch.where(unstable, phi_u, phi_s)

    def integral_m(self, zeta, zeta_0, ln_z_z0):
        c = self.c
        unstable = zeta < 0.0
        one = torch.ones_like(zeta)
        x = torch.sqrt(1 - 16.0 * torch.clamp(zeta, max=0.0))
        x_0 = torch.sqrt(1 - 16.0 * torch.clamp(zeta_0, max=0.0))
        zeta, zeta_0 = torch.where(unstable, one * 0, zeta), torch.where(unstable, one * 0, zeta_0)
        x = torch.sqrt(x)
        x_0 = torch.sqrt(x_0)
        x1 = 1.0 + x
        x1_0 = 1.0 + x_0
        num = x1 * x1 * (1.0 + x * x)
        denom = x1_0 * x1_0 * (1.0 + x_0 * x_0)
        y = torch.atan(x) - torch.atan(x_0)
        psi_u = ln_z_z0 - torch.log(num / denom) + 2 * y
        if c.stable_option == 1:
            psi_s = (ln_z_z0 + (5.0 - self.b_stab) * torch.log((1.0 + zeta) / (1.0 + zeta_0))
                     + self.b_stab * (zeta - zeta_0))
        else:
            zt = c.zeta_trans
            weak = ln_z_z0 + 5.0 * (zeta - zeta_0)
            strong = zeta > zt
            xs = ((self.lam - 1.0) * torch.log(torch.where(strong, zeta, one * zt) / zt)
                  + self.b_stab * (zeta - zt))
            s1 = ln_z_z0 + xs + 5.0 * (zt - zeta_0)
            s2 = self.lam * ln_z_z0 + self.b_stab * (zeta - zeta_0)
            psi_s = torch.where(strong, torch.where(zeta_0 <= zt, s1, s2), weak)
        return torch.where(unstable, psi_u, psi_s)

    def integral_tq(self, zeta, zeta_t, zeta_q, ln_z_zt, ln_z_zq):
        c = self.c
        unstable = zeta < 0.0
        one = torch.ones_like(zeta)
        x = torch.sqrt(1 - 16.0 * torch.clamp(zeta, max=0.0))
        x_t = torch.sqrt(1 - 16.0 * torch.clamp(zeta_t, max=0.0))
        x_q = torch.sqrt(1 - 16.0 * torch.clamp(zeta_q, max=0.0))
        zeta = torch.where(unstable, one * 0, zeta)
        zeta_t = torch.where(unstable, one * 0, zeta_t)
        zeta_q = torch.where(unstable, one * 0, zeta_q)
        pt_u = ln_z_zt - 2.0 * torch.log((1.0 + x) / (1.0 + x_t))
        pq_u = ln_z_zq - 2.0 * torch.log((1.0 + x) / (1.0 + x_q))
        if c.stable_option == 1:
            pt_s = (ln_z_zt + (5.0 - self.b_stab) * torch.log((1.0 + zeta) / (1.0 + zeta_t))
                    + self.b_stab * (zeta - zeta_t))
            pq_s = (ln_z_zq + (5.0 - self.b_stab) * torch.log((1.0 + zeta) / (1.0 + zeta_q))
                    + self.b_stab * (zeta - zeta_q))
        else:
            zt = c.zeta_trans
            strong = zeta > zt
            xs = ((self.lam - 1.0) * torch.log(torch.where(strong, zeta, one * zt) / zt)
                  + self.b_stab * (zeta - zt))
            pt_s = torch.where(strong,
                               torch.where(zeta_t <= zt, ln_z_zt + xs + 5.0 * (zt - zeta_t),
                                           self.lam * ln_z_zt + self.b_stab * (zeta - zeta_t)),
                               ln_z_zt + 5.0 * (zeta - zeta_t))
            pq_s = torch.where(strong,
                               torch.where(zeta_q <= zt, ln_z_zq + xs + 5.0 * (zt - zeta_q),
                                           self.lam * ln_z_zq + self.b_stab * (zeta - zeta_q)),
                               ln_z_zq + 5.0 * (zeta - zeta_q))
        return torch.where(unstable, pt_u, pt_s), torch.where(unstable, pq_u, pq_s)

    # ---- drag --------------------------------------------------------------------------------
    def solve_zeta(self, rich, z, z0, zt, zq, mask):
        """monin_obukhov_solve_zeta -> (f_m, f_t, f_q); only meaningful where mask."""
        z_z0 = z / z0
        z_zt = z / zt
        z_zq = z / zq
        ln_z_z0 = torch.log(z_z0)
        ln_z_zt = torch.log(z_zt)
        ln_z_zq = torch.log(z_zq)
        zero = torch.zeros_like(rich)
        zeta = torch.where(mask, rich * ln_z_z0 * ln_z_z0 / ln_z_zt, zero)
        zeta = torch.where(mask & (rich >= 0.0), zeta / (1.0 - rich / self.c.rich_crit), zeta)
        f_m = torch.zeros_like(rich)
        f_t = torch.zeros_like(rich)
        f_q = torch.zeros_like(rich)
        active = mask.clone()
        for _ in range(self.MAX_ITER):
            if not bool(active.any()):          # every column converged: later passes change nothing
                break
            small = active & (torch.abs(zeta) < self.ZETA_MIN)
            zeta = torch.where(small, zero, zeta)
            f_m = torch.where(small, ln_z_z0, f_m)
            f_t = torch.where(small, ln_z_zt, f_t)
            f_q = torch.where(small, ln_z_zq, f_q)
            active = active & ~small
            zsafe = torch.where(active, zeta, torch.ones_like(zeta))
            rzeta = 1.0 / zsafe
            zeta_0 = torch.where(active, zeta / z_z0, zero)
            zeta_t = torch.where(active, zeta / z_zt, zero)
            zeta_q = torch.where(active, zeta / z_zq, zero)
            phi_m = self.derivative_m(zsafe)
            phi_m_0 = self.derivative_m(zeta_0)
            phi_t = self.derivative_t(zsafe)
            phi_t_0 = self.derivative_t(zeta_t)
            fm_new = self.integral_m(zsafe, zeta_0, ln_z_z0)
            ft_new, fq_new = self.integral_tq(zsafe, zeta_t, zeta_q, ln_z_zt, ln_z_zq)
            f_m = torch.where(active, fm_new, f_m)
            f_t = torch.where(active, ft_new, f_t)
            f_q = torch.where(active, fq_new, f_q)
            df_m = (phi_m - phi_m_0) * rzeta
            df_t = (phi_t - phi_t_0) * rzeta
            # inactive points may hold f = 0; keep them finite so no NaN reaches the backward pass
            one = torch.ones_like(f_m)
            fm_s = torch.where(active, f_m, one)
            ft_s = torch.where(active, f_t, one)
            rich_1 = zsafe * ft_s / (fm_s * fm_s)
            d_rich = rich_1 * (rzeta + df_t / ft_s - 2.0 * df_m / fm_s)
            correction = (rich - rich_1) / torch.where(active, d_rich, one)
            corr = torch.minimum(torch.abs(correction), torch.abs(correction / zsafe))
            go = active & (corr > self.ERROR)
            zeta = torch.where(go, zeta + correction, zeta)
            active = go
        return f_m, f_t, f_q

    def drag(self, pt, pt0, z, z0, zt, zq, speed, avail=None):
        """monin_obukhov_drag_1d -> drag_m, drag_t, drag_q, u_star, b_star (zero where not avail)."""
        mask = torch.ones_like(pt, dtype=torch.bool) if avail is None else avail
        r_crit = 0.95 * self.c.rich_crit
        sqrt_drag_min = math.sqrt(self.c.drag_min) if self.c.drag_min != 0.0 else 0.0
        delta_b = GRAV * (pt0 - pt) / pt0
        rich = torch.where(mask, -z * delta_b / (speed * speed + self.SMALL), torch.zeros_like(pt))
        zz = torch.maximum(torch.maximum(torch.maximum(z, z0), zt), zq)
        zero = torch.zeros_like(pt)
        if self.c.neutral:
            fm = torch.log(zz / z0)
            ft = torch.log(zz / zt)
            fq = torch.log(zz / zq)
            us, bs, qs = VONKARM / fm, VONKARM / ft, VONKARM / fq
        else:
            mask_1 = mask & (rich < r_crit)
            mask_2 = mask & (rich >= r_crit)
            fm, ft, fq = self.solve_zeta(rich, zz, z0, zt, zq, mask_1)
            one = torch.ones_like(fm)
            us = torch.clamp(VONKARM / torch.where(mask_1, fm, one), min=sqrt_drag_min)
            bs = torch.clamp(VONKARM / torch.where(mask_1, ft, one), min=sqrt_drag_min)
            qs = torch.clamp(VONKARM / torch.where(mask_1, fq, one), min=sqrt_drag_min)
            us = torch.where(mask_2, torch.full_like(us, sqrt_drag_min), us)
            bs = torch.where(mask_2, torch.full_like(bs, sqrt_drag_min), bs)
            qs = torch.where(mask_2, torch.full_like(qs, sqrt_drag_min), qs)
        drag_m = torch.where(mask, us * us, zero)
        drag_t = torch.where(mask, us * bs, zero)
        drag_q = torch.where(mask, us * qs, zero)
        if not self.c.neutral:
            drag_m = torch.where(mask_2, torch.full_like(zero, self.c.drag_min), drag_m)
            drag_t = torch.where(mask_2, torch.full_like(zero, self.c.drag_min), drag_t)
            drag_q = torch.where(mask_2, torch.full_like(zero, self.c.drag_min), drag_q)
        u_star = torch.where(mask, us * speed, zero)
        b_star = torch.where(mask, bs * delta_b, zero)
        return drag_m, drag_t, drag_q, u_star, b_star

    # ---- profiles ----------------------------------------------------------------------------
    def profile(self, zref, zref_t, z, z0, zt, zq, u_star, b_star, q_star, avail=None):
        """monin_obukhov_profile_1d -> del_m, del_t, del_q (zero where not avail)."""
        mask = torch.ones_like(z, dtype=torch.bool) if avail is None else avail
        one = torch.ones_like(z)
        zs = torch.where(mask, z, one)
        ln_z_z0 = torch.log(zs / torch.where(mask, z0, one))
        ln_z_zt = torch.log(zs / torch.where(mask, zt, one))
        ln_z_zq = torch.log(zs / torch.where(mask, zq, one))
        ln_z_zref = torch.log(zs / zref)
        ln_z_zref_t = torch.log(zs / zref_t)
        zero = torch.zeros_like(z)
        if self.c.neutral:
            del_m = 1.0 - ln_z_zref / ln_z_z0
            del_t = 1.0 - ln_z_zref_t / ln_z_zt
            del_q = 1.0 - ln_z_zref_t / ln_z_zq
        else:
            ok = mask & (u_star > 0.0)
            us = torch.where(ok, u_star, one)
            mo_length_inv = torch.where(ok, -VONKARM * b_star / (us * us), zero)
            # Where u_star <= 0 the Fortran leaves zeta undefined (uninitialised);
            # zero is used here.
            zeta = z * mo_length_inv
            zeta_0 = z0 * mo_length_inv
            zeta_t = zt * mo_length_inv
            zeta_q = zq * mo_length_inv
            zeta_ref = zref * mo_length_inv
            zeta_ref_t = zref_t * mo_length_inv
            f_m = self.integral_m(zeta, zeta_0, ln_z_z0)
            f_m_ref = self.integral_m(zeta, zeta_ref, ln_z_zref)
            f_t, f_q = self.integral_tq(zeta, zeta_t, zeta_q, ln_z_zt, ln_z_zq)
            f_t_ref, f_q_ref = self.integral_tq(zeta, zeta_ref_t, zeta_ref_t, ln_z_zref_t, ln_z_zref_t)
            del_m = 1.0 - f_m_ref / f_m
            del_t = 1.0 - f_t_ref / f_t
            del_q = 1.0 - f_q_ref / f_q
        return (torch.where(mask, del_m, zero), torch.where(mask, del_t, zero),
                torch.where(mask, del_q, zero))

    # ---- diffusivities -----------------------------------------------------------------------
    def diff(self, z, u_star, b_star):
        """monin_obukhov_diff: k_m, k_h at heights z; u_star, b_star broadcast against z."""
        uss = torch.clamp(u_star, min=self.USTAR_MIN)
        if self.c.neutral:
            k_m = VONKARM * uss * z
            return k_m, k_m
        zeta = -VONKARM * b_star * z / (uss * uss)
        phi_m = self.derivative_m(zeta)
        phi_h = self.derivative_t(zeta)
        return VONKARM * uss * z / phi_m, VONKARM * uss * z / phi_h

    def stable_mix(self, rich):
        c = self.c
        zero = torch.zeros_like(rich)
        if c.stable_option == 1:
            ok = (rich > 0.0) & (rich < c.rich_crit)
            r = 1.0 / torch.where(ok, rich, torch.ones_like(rich))
            a = r - self.b_stab
            b = r - (1.0 + 5.0)
            cc = -1.0
            zeta = (-b + torch.sqrt(b * b - 4.0 * a * cc)) / (2.0 * a)
            phi = 1.0 + self.b_stab * zeta + (5.0 - self.b_stab) * zeta / (1.0 + zeta)
            return torch.where(ok, 1.0 / (phi * phi), zero)
        mix = torch.where((rich > 0.0) & (rich <= self.rich_trans), (1.0 - 5.0 * rich) ** 2, zero)
        mix = torch.where((rich > self.rich_trans) & (rich < c.rich_crit),
                          ((1.0 - self.b_stab * rich) / self.lam) ** 2, mix)
        return mix
