"""Solar geometry (shared/astronomy/astronomy.F90, siena_201207).

The orbit table (num_angles RK4 steps of Kepler's second law), the declination and
earth-sun distance factor, and diurnal_solar with time averaging over dt (the form
radiation_driver uses: cosz and the daylight fraction averaged over the radiation
step). Per column, vectorised with torch; the orbital quantities are scalars.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..constants import PI
from ..time_manager import ModelTime, set_date

SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class AstronomyConfig:
    ecc: float = 0.01671
    obliq: float = 23.439
    per: float = 102.932
    period: int = 0
    year_ae: int = 1998
    month_ae: int = 9
    day_ae: int = 23
    hour_ae: int = 5
    minute_ae: int = 37
    second_ae: int = 0
    num_angles: int = 3600

    @classmethod
    def from_nml(cls, nml: dict) -> "AstronomyConfig":
        g = nml.get("astronomy_nml", {})
        return cls(**{k: type(getattr(cls, k))(g[k]) for k in cls.__dataclass_fields__ if k in g})


class Astronomy:
    def __init__(self, config: AstronomyConfig = AstronomyConfig()):
        c = self.c = config
        self.twopi = 2.0 * PI
        self.deg_to_rad = self.twopi / 360.0
        self.ae_ref = set_date(c.year_ae, c.month_ae, c.day_ae, c.hour_ae, c.minute_ae, c.second_ae)
        self.period = c.period if c.period != 0 else int(SECONDS_PER_DAY * 365 + 6 * 3600)
        self._orbit()

    def r_inv_squared(self, ang):
        c = self.c
        rad_per = c.per * self.deg_to_rad
        r = (1 - c.ecc ** 2) / (1.0 + c.ecc * math.cos(ang - rad_per))
        return r ** (-2)

    def _orbit(self):
        n = self.c.num_angles
        orb = np.zeros(n + 1)
        dt = self.twopi / float(n)
        dt = dt * math.sqrt(1.0 - self.c.ecc ** 2)
        f = self.r_inv_squared
        for i in range(1, n + 1):
            a = orb[i - 1]
            d1 = dt * f(a)
            d2 = dt * f(a + 0.5 * d1)
            d3 = dt * f(a + 0.5 * d2)
            d4 = dt * f(a + d3)
            d5 = d1 / 6.0 + d2 / 3.0 + d3 / 3.0 + d4 / 6.0
            orb[i] = a + d5
        self.orb_angle = orb

    def angle(self, t):
        n = self.c.num_angles
        norm_time = t * float(n) / self.twopi
        i = int(math.floor(norm_time)) % n
        x = norm_time - math.floor(norm_time)
        a = (1.0 - x) * self.orb_angle[i] + x * self.orb_angle[i + 1]
        return math.fmod(a, self.twopi) if a >= 0 else a % self.twopi

    def declination(self, ang):
        return math.asin(-math.sin(self.c.obliq * self.deg_to_rad) * math.sin(ang))

    def universal_time(self, time: ModelTime) -> float:
        return self.twopi * float(time.seconds) / SECONDS_PER_DAY

    def orbital_time(self, time: ModelTime) -> float:
        d = abs((time.days * 86400 + time.seconds) - (self.ae_ref.days * 86400 + self.ae_ref.seconds))
        t = float(d) / float(self.period)
        t = self.twopi * (t - math.floor(t))
        if (time.days, time.seconds) < (self.ae_ref.days, self.ae_ref.seconds):
            t = self.twopi - t
        return t

    @staticmethod
    def half_day(lat, dec):
        eps = 1.0e-05
        lat = torch.where(lat == 0.5 * PI, lat - eps, torch.where(lat == -0.5 * PI, lat + eps, lat))
        chd = -torch.tan(lat) * math.tan(dec)
        h = torch.where(chd <= -1.0, torch.full_like(lat, PI), torch.zeros_like(lat))
        mid = (chd > -1.0) & (chd < 1.0)
        return torch.where(mid, torch.acos(torch.clamp(chd, -1.0, 1.0)), h)

    def diurnal_solar(self, lat, lon, time: ModelTime, dt_seconds: int | None = None):
        """diurnal_solar_cal: -> (cosz, fracday, rrsun). lat/lon (ncol,) radians."""
        gmt = self.universal_time(time)
        tae = self.orbital_time(time)
        ang = self.angle(tae)
        dec = self.declination(ang)
        rrsun = self.r_inv_squared(ang)
        aa = torch.sin(lat) * math.sin(dec)
        bb = torch.cos(lat) * math.cos(dec)
        t = gmt + lon - PI
        t = torch.where(t >= PI, t - self.twopi, t)
        t = torch.where(t < -PI, t + self.twopi, t)
        h = self.half_day(lat, dec)
        twopi = self.twopi
        zero = torch.zeros_like(t)
        if dt_seconds is None:
            day = torch.abs(t) < h
            cosz = torch.where(day, aa + bb * torch.cos(t), zero)
            fracday = torch.where(day, torch.ones_like(t), zero)
            return torch.clamp(cosz, min=0.0), fracday, rrsun
        dt = self.universal_time(ModelTime(0, int(dt_seconds)))
        tt = t + dt
        st, stt, sh = torch.sin(t), torch.sin(tt), torch.sin(h)
        one = torch.ones_like(t)
        safe = lambda d: torch.where(d != 0.0, d, one)
        cosz = zero
        cosz = torch.where((tt + h != 0.0) & (t < -h) & (torch.abs(tt) <= h), aa + bb * (stt + sh) / safe(tt + h), cosz)
        cosz = torch.where((t < -h) & (h != 0.0) & (h < tt), aa + bb * (sh + sh) / safe(h + h), cosz)
        cosz = torch.where((torch.abs(t) <= h) & (torch.abs(tt) <= h), aa + bb * (stt - st) / safe(tt - t), cosz)
        cosz = torch.where((h - t != 0.0) & (torch.abs(t) <= h) & (h < tt), aa + bb * (sh - st) / safe(h - t), cosz)
        m = (twopi - h < tt) & (tt + h - twopi != 0.0) & (t <= h)
        cosz = torch.where(m, (cosz * (h - t) + (aa * (tt + h - twopi) + bb * (stt + sh)))
                           / safe((h - t) + (tt + h - twopi)), cosz)
        cosz = torch.where((h < t) & (twopi - h >= tt), zero, cosz)
        cosz = torch.where((h < t) & (twopi - h < tt), aa + bb * (stt + sh) / safe(tt + h - twopi), cosz)
        fracday = zero
        fracday = torch.where((t < -h) & (tt < -h), zero, fracday)
        fracday = torch.where((t < -h) & (torch.abs(tt) <= h), (tt + h) / dt, fracday)
        fracday = torch.where((t < -h) & (h < tt), (h + h) / dt, fracday)
        fracday = torch.where((torch.abs(t) <= h) & (torch.abs(tt) <= h), (tt - t) / dt, fracday)
        fracday = torch.where((torch.abs(t) <= h) & (h < tt), (h - t) / dt, fracday)
        fracday = torch.where(h < t, zero, fracday)
        fracday = torch.where(twopi - h < tt, fracday + (tt + h - twopi) / dt, fracday)
        return torch.clamp(cosz, min=0.0), fracday, rrsun
