"""The sea_esf radiation package (sea_esf_rad.F90 and the radiation_calc part of
radiation_driver.F90, siena_201207) for the AM2.1 configuration: radiative gases and
Fortuin-Kelder ozone, the Shettle-Fenn aerosol climatology, 'strat' clouds with predicted
microphysics and random overlap, the ESF shortwave and the sealw99 longwave.

SeaEsfRad(...) is the ``package`` of radiation.driver.RadiationDriver:
    package(atmos, surface, astro, time, do_lw, do_sw, extra) -> dict
where ``extra`` carries the cloud tracers (ql, qi, qa) of the radiation step and
``surface`` the land flag and the four surface albedos.

The time-independent set-up (spectral data, tables, gas transmission functions on the
reference pressures) is built once in the constructor, in float64 numpy; the per-call
work is torch, on the dtype/device of the inputs.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .aerosol import AerosolClimatology, AerosolOptics
from .cloud_optics import CloudLWOptics, CloudSWOptics
from .clouds import CloudRadConfig, cloud_spec_strat
from .esfsw import ShortwaveESF
from .esfsw_data import lean_solar, read_solar_spectrum
from .gases import RadiativeGases
from .lw_gases_stdtf import LblTransmission
from .lw_tables import LongwaveData, ptz
from .sealw99 import SeaLW99


def _check_config(nml: dict):
    """The package supports the AM2.1 SCM configuration only; refuse anything else loudly."""
    lw = nml.get("sealw99_nml", {})
    want = dict(continuum_form="ckd2.1", linecatalog_form="hitran_2000")
    for k, v in want.items():
        if str(lw.get(k, "")).strip() != v:
            raise NotImplementedError(f"sealw99_nml {k} = {lw.get(k)!r} (supported: {v!r})")
    if not bool(lw.get("do_lwcldemiss", False)):
        raise NotImplementedError("sealw99_nml do_lwcldemiss = .false.")
    if bool(lw.get("do_thick", False)) or bool(lw.get("do_nlte", False)):
        raise NotImplementedError("sealw99_nml do_thick / do_nlte")
    if int(lw.get("no_h2o_bands_1200_1400", 1)) != 1:
        raise NotImplementedError("sealw99_nml no_h2o_bands_1200_1400 /= 1")
    if int(nml.get("lw_gases_stdtf_nml", {}).get("NSTDCO2LVLS", 496)) != 496:
        raise NotImplementedError("NSTDCO2LVLS /= 496")
    mp = nml.get("microphys_rad_nml", {})
    if str(mp.get("lwem_form", "fuliou")).strip() != "fuliou":
        raise NotImplementedError("microphys_rad_nml lwem_form")
    ae = nml.get("aerosolrad_package_nml", {})
    if not (bool(ae.get("do_lwaerosol", False)) and bool(ae.get("do_swaerosol", False))):
        raise NotImplementedError("aerosols must be active in both the SW and the LW")


class SeaEsfRad:
    def __init__(self, nml: dict, data_dir: str, boxes, pref: np.ndarray, do_totcld_forcing: bool = True):
        """boxes: [((lon0, lon1), (lat0, lat1))] per column [radians] for the climatologies;
        pref: (nlev+1, 2) reference pressures [Pa] (pfull for psurf 101325 and 81060 Pa, then
        the two surface pressures) as physics_driver_init receives them."""
        _check_config(nml)
        self.gases = RadiativeGases(nml, data_dir, boxes)
        self.aerosol = AerosolClimatology(nml, data_dir, boxes)
        spec = read_solar_spectrum(data_dir, nml)
        self.aero_optics = AerosolOptics(nml, data_dir, spec, self.aerosol.names)
        self.cloud_cfg = CloudRadConfig.from_nml(nml)
        self.cloud_sw = CloudSWOptics(spec)
        self.cloud_lw = CloudLWOptics()
        self.sw = ShortwaveESF(spec, do_totcld_forcing=do_totcld_forcing,
                               lean=lean_solar(data_dir, nml, spec.nbands))
        pref = np.asarray(pref, float)
        stemp, gtemp, ix = ptz(pref[:, 0])
        rrv = self.gases.rrv
        gas_tf = LblTransmission(data_dir, pref).tables(rrv["co2"], rrv["ch4"], rrv["n2o"])
        gas_tf.update(stemp=stemp, gtemp=gtemp, ixprkminh2o=ix)
        self.lw = SeaLW99(LongwaveData(data_dir), gas_tf, do_totcld_forcing=do_totcld_forcing)

    def __call__(self, atm: dict, surface: dict, astro: dict, time, do_lw: bool, do_sw: bool, extra=None):
        dt, dev = atm["temp"].dtype, atm["temp"].device
        pflux = atm["pflux"]
        qo3 = self.gases.qo3(time, pflux).to(dt)
        aer = self.aerosol(time, pflux).to(dt)
        land = surface["land"]
        cs = cloud_spec_strat(self.cloud_cfg, extra["ql"], extra["qi"], extra["qa"], atm["press"], pflux,
                              atm["cloudtemp"], land)
        out = dict(cld_spec=cs, qo3=qo3, aerosol=aer)
        rrv = self.gases.rrv
        if do_sw:
            aero_sw = self.aero_optics.sw_optical_depths(aer, atm["aerosolrelhum"])
            ext, sct, asy = self.cloud_sw(cs["conc_drop"], cs["size_drop"], cs["conc_ice"], cs["size_ice"])
            cld = dict(camtsw=cs["camtsw"], cldext=ext, cldsct=sct, cldasymm=asy)
            surf = {k: surface[k] for k in ("asfc_vis_dir", "asfc_nir_dir", "asfc_vis_dif", "asfc_nir_dif")}
            out.update(self.sw(atm, qo3, rrv["co2"], aero_sw, cld, surf, astro))
        if do_lw:
            tot, ke = self.aero_optics.lw_optical_depths(aer, atm["aerosolrelhum"])
            _, em = self.cloud_lw(cs["conc_drop"], cs["conc_ice"], cs["size_ice"], atm["clouddeltaz"])
            cld = dict(cmxolw=cs["cmxolw"], crndlw=cs["crndlw"], emmxolw=em, emrndlw=em)
            gases = {"rrv" + g: rrv[g] for g in ("f11", "f12", "f113", "f22")}
            out.update(self.lw(atm, qo3, gases, tot, ke, cld))
        return out
