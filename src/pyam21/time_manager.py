"""Model time on the FMS Julian calendar (shared/time_manager/time_manager.F90).

A ModelTime is (days, seconds) since 0001-01-01 00:00, exactly as FMS stores a
time_type with calendar = 'julian' (every fourth year is a leap year). Only the
Julian calendar is used by the SCM cases.
"""
from __future__ import annotations

from dataclasses import dataclass

DAYS_PER_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


@dataclass(frozen=True, order=True)
class ModelTime:
    days: int
    seconds: int = 0

    def __post_init__(self):
        d, s = divmod(self.seconds, 86400)
        object.__setattr__(self, "days", self.days + d)
        object.__setattr__(self, "seconds", s)

    def __add__(self, other: "ModelTime") -> "ModelTime":
        return ModelTime(self.days + other.days, self.seconds + other.seconds)

    def __sub__(self, other: "ModelTime") -> "ModelTime":
        return ModelTime(self.days - other.days, self.seconds - other.seconds)

    @property
    def total_seconds(self) -> int:
        return self.days * 86400 + self.seconds

    def date(self):
        return get_date(self)


def set_time(seconds: int, days: int = 0) -> ModelTime:
    return ModelTime(days, seconds)


def set_date(year, month, day, hour=0, minute=0, second=0) -> ModelTime:
    """set_date_julian_private."""
    leap = year % 4 == 0
    nleapyr = (year - 1) // 4
    if month != 2 and day > DAYS_PER_MONTH[month - 1]:
        raise ValueError("invalid date")
    if month == 2 and (day > 29 or (not leap and day > 28)):
        raise ValueError("invalid date")
    ndays = 0
    for m in range(1, month):
        ndays += DAYS_PER_MONTH[m - 1]
        if leap and m == 2:
            ndays += 1
    seconds = second + 60 * (minute + 60 * hour)
    days = day - 1 + ndays + 365 * (year - nleapyr - 1) + 366 * nleapyr
    return ModelTime(days, seconds)


def get_date(t: ModelTime):
    """get_date_julian_private -> (year, month, day, hour, minute, second)."""
    nfour = t.days // (4 * 365 + 1)
    day = t.days % (4 * 365 + 1)
    nex = day // 365
    if nex == 4:
        nex, day = 3, 366
    else:
        day = day % 365 + 1
    leap = nex == 3
    year = 1 + 4 * nfour + nex
    month = 12
    for m in range(1, 13):
        month = m
        dtm = 29 if (leap and m == 2) else DAYS_PER_MONTH[m - 1]
        if day <= dtm:
            break
        day -= dtm
    s = t.seconds
    hour = s // 3600
    s -= hour * 3600
    minute = s // 60
    return year, month, day, hour, minute, s - 60 * minute


def days_in_year(t: ModelTime) -> int:
    return 366 if get_date(t)[0] % 4 == 0 else 365


def days_in_month(t: ModelTime) -> int:
    y, m = get_date(t)[:2]
    return 29 if (m == 2 and y % 4 == 0) else DAYS_PER_MONTH[m - 1]
