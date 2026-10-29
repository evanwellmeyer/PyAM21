"""Saturation vapor pressure lookup tables of FMS.

Translation of shared/sat_vapor_pres/sat_vapor_pres.F90 and sat_vapor_pres_k.F90
(siena_201207):
  - three tables, each with first and second derivatives, from -160 C to 100 C at
    0.1 K resolution:  TABLE  (ice below -20 C, water above 0 C, linear blend
    between; the default "es"),  TABLE2 (over liquid water only) and TABLE3 (over
    ice below 0 C, water above).  TABLE2/3 exist only if the namelist asks for them;
    here they are always built (building them has no side effects).
  - lookup is a quadratic Taylor expansion about the table node below
    temp + 0.05 K:  es = T(i) + del*(D(i) + del*D2(i)),  des = D(i) + 2*del*D2(i).

Autograd: d(es)/dT of the lookup is exactly the table derivative des, so
gradients are consistent with the Fortran tangent (dqsdT etc.).

Out-of-range temperatures are an error in the Fortran (FATAL, or -999 results).
Here the index is clamped so the computation stays finite on any device;
``count_bad`` reports how many values are out of range (it syncs the device).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .constants import HLV, RDGAS, RVGAS, TFREEZE, ES0
from .core import TableCache, fortran_int

EPSILO = RDGAS / RVGAS
ZVIR = RVGAS / RDGAS - 1.0

_ESBASW = 101324.60
_ESBASI = 610.71


def _es_ice(t: float) -> float:
    tbasi = TFREEZE
    x = (-9.09718 * (tbasi / t - 1.0) - 3.56654 * math.log10(tbasi / t)
         + 0.876793 * (1.0 - t / tbasi) + math.log10(_ESBASI))
    return 10.0 ** x


def _es_h2o(t: float) -> float:
    tbasw = TFREEZE + 100.0
    x = (-7.90298 * (tbasw / t - 1) + 5.02808 * math.log10(tbasw / t)
         - 1.3816e-07 * (10.0 ** ((1 - t / tbasw) * 11.344) - 1)
         + 8.1328e-03 * (10.0 ** ((tbasw / t - 1) * (-3.49149)) - 1)
         + math.log10(_ESBASW))
    return 10.0 ** x


def compute_es(t: float) -> float:
    """compute_es_k: ice below -20 C, water above 0 C, linear blend between."""
    tbasi = TFREEZE
    esice = _es_ice(t) if t < tbasi else 0.0
    esh2o = _es_h2o(t) if t > -20.0 + tbasi else 0.0
    if t <= -20.0 + tbasi:
        return esice
    if t >= tbasi:
        return esh2o
    return 0.05 * ((tbasi - t) * esice + (t - tbasi + 20.0) * esh2o)


def compute_es_liq(t: float) -> float:
    return _es_h2o(t)


def compute_es_liq_ice(t: float) -> float:
    return _es_ice(t) if t < TFREEZE else _es_h2o(t)


@dataclass(frozen=True)
class SatVaporPresConfig:
    """sat_vapor_pres_nml (the values that change results) and the table range."""
    use_exact_qs: bool = False
    do_simple: bool = False
    tcmin: int = -160
    tcmax: int = 100
    esres: int = 10


class SatVaporPres:
    """The three FMS tables and their lookups."""

    def __init__(self, config: SatVaporPresConfig = SatVaporPresConfig()):
        self.config = config
        tcmin, tcmax = (config.tcmin, config.tcmax) if not config.do_simple else (-350, 350)
        nsize = (tcmax - tcmin) * config.esres + 1
        self.table_size = nsize
        self.dtres = (float(tcmax) - float(tcmin)) / (nsize - 1)
        self.tmin = float(tcmin) + TFREEZE
        self.dtinv = 1.0 / self.dtres
        self.teps = 0.5 * self.dtres
        tinrc = 0.1 * self.dtres
        tfact = 5 * self.dtinv

        def build(fn):
            tab = np.empty(nsize)
            dtab = np.empty(nsize)
            for i in range(nsize):
                tem = self.tmin + self.dtres * float(i)
                tab[i] = fn(tem)
                dtab[i] = (fn(tem + tinrc) - fn(tem - tinrc)) * tfact
            d2 = np.empty(nsize)
            d2[1:-1] = 0.25 * self.dtinv * (dtab[2:] - dtab[:-2])
            d2[0] = 0.50 * self.dtinv * (dtab[1] - dtab[0])
            d2[-1] = 0.50 * self.dtinv * (dtab[-1] - dtab[-2])
            return tab, dtab, d2

        if config.do_simple:
            tab = np.empty(nsize)
            dtab = np.empty(nsize)
            for i in range(nsize):
                tem = self.tmin + self.dtres * float(i)
                tab[i] = ES0 * 610.78 * math.exp(-HLV / RVGAS * (1.0 / tem - 1.0 / TFREEZE))
                dtab[i] = HLV * tab[i] / RVGAS / tem ** 2.0
            d2 = np.empty(nsize)
            d2[1:-1] = 0.25 * self.dtinv * (dtab[2:] - dtab[:-2])
            d2[0] = 0.50 * self.dtinv * (dtab[1] - dtab[0])
            d2[-1] = 0.50 * self.dtinv * (dtab[-1] - dtab[-2])
            mixed = (tab, dtab, d2)
        else:
            mixed = build(compute_es)
        self.tables = {
            "mixed": tuple(TableCache(a) for a in mixed),
            "liq": tuple(TableCache(a) for a in build(compute_es_liq)),
            "liq_ice": tuple(TableCache(a) for a in build(compute_es_liq_ice)),
        }

    # ---- lookups -------------------------------------------------------------------------
    def _index(self, temp: torch.Tensor):
        tmp = temp - self.tmin
        ind = fortran_int(self.dtinv * (tmp + self.teps))
        ind = ind.clamp(0, self.table_size - 1)
        # del uses the (clamped) integer node, as the Fortran does for in-range values
        dl = tmp - self.dtres * ind.to(temp.dtype)
        return ind, dl

    def count_bad(self, temp: torch.Tensor) -> int:
        ind = fortran_int(self.dtinv * (temp - self.tmin + self.teps))
        return int(((ind < 0) | (ind >= self.table_size)).sum())

    def lookup_es(self, temp: torch.Tensor, which: str = "mixed") -> torch.Tensor:
        t, d, d2 = (c.like(temp) for c in self.tables[which])
        ind, dl = self._index(temp)
        return t[ind] + dl * (d[ind] + dl * d2[ind])

    def lookup_des(self, temp: torch.Tensor, which: str = "mixed") -> torch.Tensor:
        _, d, d2 = (c.like(temp) for c in self.tables[which])
        ind, dl = self._index(temp)
        return d[ind] + 2.0 * dl * d2[ind]

    def lookup_es_des(self, temp: torch.Tensor, which: str = "mixed"):
        t, d, d2 = (c.like(temp) for c in self.tables[which])
        ind, dl = self._index(temp)
        return t[ind] + dl * (d[ind] + dl * d2[ind]), d[ind] + 2.0 * dl * d2[ind]

    # ---- specific humidity / mixing ratio at saturation ----------------------------------
    def compute_qs(self, temp, press, q=None, hc: float = 1.0, which: str = "mixed",
                   want_dqsdT: bool = False, want_esat: bool = False,
                   eps: float = EPSILO, zvir: float = ZVIR):
        """compute_qs_k. Returns qs, or (qs, dqsdT[, esat]) as requested.

        which: 'mixed' (default table), 'liq' (es_over_liq) or 'liq_ice'
        (es_over_liq_and_ice).
        """
        if want_dqsdT:
            esloc, desat = self.lookup_es_des(temp, which)
            desat = desat * hc
        else:
            esloc = self.lookup_es(temp, which)
            desat = None
        esloc = esloc * hc
        if q is not None and self.config.use_exact_qs:
            qs = (1.0 + zvir * q) * eps * esloc / press
            dqsdT = (1.0 + zvir * q) * eps * desat / press if want_dqsdT else None
        else:
            denom = press - (1.0 - eps) * esloc
            pos = denom > 0.0
            qs = torch.where(pos, eps * esloc / torch.where(pos, denom, torch.ones_like(denom)),
                             torch.full_like(denom, eps))
            dqsdT = eps * press * desat / denom ** 2 if want_dqsdT else None
        out = [qs]
        if want_dqsdT:
            out.append(dqsdT)
        if want_esat:
            out.append(esloc)
        return out[0] if len(out) == 1 else tuple(out)

    def compute_mrs(self, temp, press, mr=None, hc: float = 1.0, which: str = "mixed",
                    want_dmrsdT: bool = False, want_esat: bool = False,
                    eps: float = EPSILO, zvir: float = ZVIR):
        """compute_mrs_k (saturation mixing ratio)."""
        raise NotImplementedError("compute_mrs is ported when a scheme needs it")


_default: SatVaporPres | None = None


def get_sat_vapor_pres() -> SatVaporPres:
    """The process-wide tables (sat_vapor_pres_init with the default namelist)."""
    global _default
    if _default is None:
        _default = SatVaporPres()
    return _default


def set_sat_vapor_pres(config: SatVaporPresConfig) -> SatVaporPres:
    global _default
    _default = SatVaporPres(config)
    return _default
