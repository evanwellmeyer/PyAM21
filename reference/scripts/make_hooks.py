"""Insert ref_dump hook calls into copies of reference source files.

Writes reference/hooks/<path> for every file patched below, starting from
reference/overrides/<path> if that exists, else reference/src/<path>.
build_reference.sh prefers hooks/ over overrides/ over src/.

Each patch is (anchor, text, where): the anchor must occur exactly once in the
file; text is inserted on its own lines before or after the line containing it.
The hooks only write when REF_DUMP_DIR is set; otherwise they do nothing, and the
model's results are unchanged (checked by comparing runs with and without them).
"""
import os
import sys

REF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dump_block(tag, items):
    """items: list of (name, fortran expression)."""
    lines = [f"      call ref_dump_begin('{tag}')"]
    for name, expr in items:
        lines.append(f"      call ref_dump('{name}', {expr})")
    lines.append("      call ref_dump_end()")
    return "\n".join(lines)


USE = "use ref_dump_mod, only: ref_dump_begin, ref_dump_end, ref_dump, ref_dump_set_step, ref_dump_step, ref_dump_active"

STATE = [("T", "Var%T"), ("u", "Var%u"), ("v", "Var%v"), ("R", "Var%R"), ("ps", "Var%ps")]

PATCHES = {
    "atmos_scm/driver/coupled/atmosphere.F90": [
        ("use scm_core_driver_mod, only:   scm_core_driver_init, &", USE, "before"),
        # --- step counter and SCM forcing -------------------------------------------------
        ("   call scm_core_driver (Time, Time_step_atmos, Hgrid, Vgrid, pdamp, fisl,    &",
         "      call get_time(Time, rd_sec, rd_day)\n"
         + dump_block("scm_forc_in", STATE + [("day", "rd_day"), ("sec", "rd_sec"),
                                              ("pfull", "pfull"), ("phalf", "phalf"),
                                              ("zfull", "zfull"), ("zhalf", "zhalf"),
                                              ("fisl", "fisl"), ("lat", "Hgrid%Tmp%aph"),
                                              ("lon", "Hgrid%Tmp%alm")]),
         "before"),
        ("                         Var, Var_dt, Forc_tend, ztend )",
         dump_block("scm_forc_out", STATE + [("Var_dt_T", "Var_dt%T"), ("Var_dt_u", "Var_dt%u"),
                                             ("Var_dt_v", "Var_dt%v"), ("Var_dt_R", "Var_dt%R"),
                                             ("omega", "omega"), ("pfull", "pfull"), ("phalf", "phalf"),
                                             ("zfull", "zfull"), ("zhalf", "zhalf"), ("ztend", "ztend")]),
         "after"),
        # --- end of step: tendencies before and after the atmosphere_up adjustments ------
        ("   Phys_tend%T = Var_dt%T - Forc_tend%T",
         dump_block("atmos_up_in", STATE + [("Var_dt_T", "Var_dt%T"), ("Var_dt_u", "Var_dt%u"),
                                            ("Var_dt_v", "Var_dt%v"), ("Var_dt_R", "Var_dt%R"),
                                            ("pfull", "pfull"), ("pdamp", "pdamp"),
                                            ("dt_atmos", "dt_atmos")]),
         "before"),
        ("     call prog_var_time_diff (dt_atmos, Var_dt, Var)",
         dump_block("atmos_up_out", STATE + [("damp_t", "damp_t"), ("damp_q", "damp_q"),
                                             ("fill_q", "fill_q")]),
         "after"),
    ],
}


SURF_DIFF = [("sd_dtmass", "Surf_diff%dtmass"), ("sd_dflux_t", "Surf_diff%dflux_t"),
             ("sd_delta_t", "Surf_diff%delta_t"), ("sd_delta_u", "Surf_diff%delta_u"),
             ("sd_delta_v", "Surf_diff%delta_v"), ("sd_dflux_tr", "Surf_diff%dflux_tr"),
             ("sd_delta_tr", "Surf_diff%delta_tr")]
TEND = [("udt", "udt"), ("vdt", "vdt"), ("tdt", "tdt"), ("qdt", "qdt"), ("rdt", "rdt")]
PREV = [("um", "um"), ("vm", "vm"), ("tm", "tm"), ("qm", "qm"), ("rm", "rm")]
CUR = [("u", "u"), ("v", "v"), ("t", "t"), ("q", "q"), ("r", "r")]
GRID = [("p_half", "p_half"), ("p_full", "p_full"), ("z_half", "z_half"), ("z_full", "z_full")]

PATCHES["atmos_param/physics_driver/physics_driver.F90"] = [
    ("use time_manager_mod,        only: time_type, get_time, operator (-), &", USE, "before"),
    ("      call damping_driver (is, js, lat, Time_next, dt,           &",
     dump_block("damping_in", [("lat", "lat"), ("dt", "dt"), ("z_pbl", "z_pbl")] + GRID + PREV + TEND),
     "before"),
    ("                           z_pbl , mask=mask, kbot=kbot)",
     dump_block("damping_out", TEND), "after"),
    ("      call vert_turb_driver (is, js, Time, Time_next, dt,            &",
     dump_block("vturb_in", [("dt", "dt"), ("tdtlw", "lw_tendency(is:ie,js:je,:)"),
                             ("frac_land", "frac_land"), ("u_star", "u_star"), ("b_star", "b_star"),
                             ("q_star", "q_star"), ("rough_mom", "rough_mom"), ("lat", "lat"),
                             ("convect", "convect(is:ie,js:je)")] + GRID + CUR + PREV + TEND),
     "before"),
    ("                             mask=mask, kbot=kbot             )",
     dump_block("vturb_out", [("diff_t_vert", "diff_t_vert"), ("diff_m_vert", "diff_m_vert"),
                              ("gust", "gust"), ("z_pbl", "z_pbl")]), "after"),
    ("      call vert_diff_driver_down (is, js, Time_next, dt, p_half,   &",
     dump_block("vdiff_down_in", [("dt", "dt"), ("diff_m", "diff_m(is:ie,js:je,:)"),
                                  ("diff_t", "diff_t(is:ie,js:je,:)"), ("dtau_du", "dtau_du"),
                                  ("dtau_dv", "dtau_dv"), ("tau_x", "tau_x"), ("tau_y", "tau_y"),
                                  ("radturbten", "radturbten(is:ie,js:je,:)")]
                + GRID + PREV + TEND + SURF_DIFF),
     "before"),
    ("                                  mask=mask, kbot=kbot           )",
     dump_block("vdiff_down_out", [("tau_x", "tau_x"), ("tau_y", "tau_y")] + TEND + SURF_DIFF), "after"),
    ("      call vert_diff_driver_up (is, js, Time_next, dt, p_half,   &",
     dump_block("vdiff_up_in", [("dt", "dt"), ("p_half", "p_half")] + TEND + SURF_DIFF), "before"),
    ("                                Surf_diff, tdt, qdt, rdt, mask=mask,  &\n                                kbot=kbot)",
     dump_block("vdiff_up_out", TEND), "after"),
]

PATCHES["atmos_param/physics_driver/physics_driver.F90"] += [
    ("      call define_surface (is, ie, js, je, albedo, albedo_vis_dir,  &",
     dump_block("rad_in", [("lat", "lat"), ("lon", "lon"), ("t_surf_rad", "t_surf_rad"), ("albedo", "albedo"),
                           ("albedo_vis_dir", "albedo_vis_dir"), ("albedo_nir_dir", "albedo_nir_dir"),
                           ("albedo_vis_dif", "albedo_vis_dif"), ("albedo_nir_dif", "albedo_nir_dif"),
                           ("frac_land", "frac_land"), ("dt", "dt")] + GRID + CUR),
     "before"),
    ("                             Radiation=Radiation, mask=mask, kbot=kbot)",
     dump_block("rad_out", [("tdt_rad", "Radiation%tdt_rad(:,:,:,1)"), ("tdtlw", "Radiation%tdtlw"),
                            ("flux_sw_surf", "Radiation%flux_sw_surf(:,:,1)"),
                            ("flux_lw_surf", "Radiation%flux_lw_surf"),
                            ("coszen", "Radiation%coszen_angle"), ("flxnet", "Radiation%flxnet"),
                            ("ufsw", "Radiation%ufsw(:,:,:,1)"), ("dfsw", "Radiation%dfsw(:,:,:,1)"),
                            ("flux_sw_vis", "Radiation%flux_sw_vis(:,:,1)"),
                            ("flux_sw_surf_dir", "Radiation%flux_sw_surf_dir(:,:,1)"),
                            ("flux_sw_surf_dif", "Radiation%flux_sw_surf_dif(:,:,1)"),
                            ("flux_sw_down_vis_dir", "Radiation%flux_sw_down_vis_dir(:,:,1)"),
                            ("flux_sw_down_vis_dif", "Radiation%flux_sw_down_vis_dif(:,:,1)"),
                            ("flux_sw_down_total_dir", "Radiation%flux_sw_down_total_dir(:,:,1)"),
                            ("flux_sw_down_total_dif", "Radiation%flux_sw_down_total_dif(:,:,1)")]),
     "after"),
]

AI = ["press", "temp", "rh2o", "zfull", "pflux", "tflux", "deltaz", "phalf", "rel_hum", "cloudtemp",
      "clouddeltaz", "cloudvapor", "aerosoltemp", "aerosolvapor", "aerosolpress", "aerosolrelhum", "tsfc", "psfc"]
CS = ["camtsw", "cmxolw", "crndlw", "ncldsw", "nmxolw", "nrndlw", "lwp", "iwp", "reff_liq", "reff_ice",
      "reff_liq_lim", "reff_ice_lim", "liq_frac", "cloud_water", "cloud_ice", "cloud_area", "cloud_droplet",
      "hi_cloud", "mid_cloud", "low_cloud", "ice_cloud"]


SCALARS = {"do_rad", "rrsun", "rrsun2", "g_rrvco2", "rrvch4", "rrvn2o", "rrvco2", "rrvf11", "rrvf12", "rrvf113",
           "rrvf22"}


def rd_block(tag, items, cond):
    """Like dump_block, inside `if (cond)`, each derived-type pointer guarded by associated()."""
    lines = [f"      if ({cond}) then", f"        call ref_dump_begin('{tag}')"]
    for name, expr in items:
        if "%" in expr and name not in SCALARS:
            head, last = expr.rsplit("%", 1)
            base = head + "%" + last.split("(")[0]
            lines.append(f"        if (associated({base})) call ref_dump('{name}', {expr})")
        else:
            lines.append(f"        call ref_dump('{name}', {expr})")
    lines += ["        call ref_dump_end()", "      endif"]
    return "\n".join(lines)


PATCHES["atmos_param/radiation_driver/radiation_driver.F90"] = [
    ("use time_manager_mod,      only: time_type, set_date, set_time,  &", USE, "before"),
    ("      call mpp_clock_begin (clouds_clock)",
     rd_block("rd_astro", [("do_rad", "do_rad"), ("solar", "Astro%solar"), ("cosz", "Astro%cosz"),
                           ("fracday", "Astro%fracday"), ("rrsun", "Astro%rrsun"),
                           ("solar2", "Astro2%solar"), ("cosz2", "Astro2%cosz"),
                           ("fracday2", "Astro2%fracday"), ("rrsun2", "Astro2%rrsun"), ("lat", "lat"),
                           ("lon", "lon")],
              "associated(Astro%cosz) .and. associated(Astro2%cosz)"), "before"),
    ("        call radiation_calc (is, ie, js, je, Rad_time, Time_next, lat, &",
     rd_block("rd_in", [(f"ai_{n}", f"Atmos_input%{n}") for n in AI]
              + [("g_rrvco2", "Atmos_input%g_rrvco2")]
              + [(f"sf_{n}", f"Surface%{n}") for n in ("asfc", "land", "asfc_vis_dir", "asfc_nir_dir",
                                                       "asfc_vis_dif", "asfc_nir_dif")]
              + [("qo3", "Rad_gases%qo3")] + [(n, f"Rad_gases%{n}") for n in
                  ("rrvch4", "rrvn2o", "rrvco2", "rrvf11", "rrvf12", "rrvf113", "rrvf22")]
              + [("aerosol", "Aerosol%aerosol")]
              + [(f"ap_{n}", f"Aerosol_props%{n}") for n in ("sw_ext", "sw_ssa", "sw_asy", "lw_ext",
                                                             "lw_ssa", "lw_asy", "sulfate_index")]
              + [(f"cr_{n}", f"Cldrad_props%{n}(:,:,:,:,1)") for n in ("cldext", "cldsct", "cldasymm",
                                                                       "abscoeff", "cldemiss", "emmxolw",
                                                                       "emrndlw")]
              + [(f"cs_{n}", f"Cld_spec%{n}") for n in CS]
              + [(f"lsc_{n}", f"Lsc_microphys%{n}") for n in ("conc_drop", "conc_ice", "size_drop",
                                                             "size_ice", "cldamt", "droplet_number")],
              "do_rad"), "before"),
    ("                                flux_ratio)",
     rd_block("rd_out", [("do_rad", "do_rad"), ("flux_ratio", "flux_ratio")]
              + [(f"ro_{n}", f"Rad_output%{n}") for n in ("tdt_rad", "tdtsw", "tdtlw", "flxnet", "ufsw", "dfsw",
                                                         "flux_sw_surf", "flux_sw_surf_dir", "flux_sw_surf_dif",
                                                         "flux_sw_down_vis_dir", "flux_sw_down_vis_dif",
                                                         "flux_sw_down_total_dir", "flux_sw_down_total_dif",
                                                         "flux_sw_vis", "flux_lw_surf", "coszen_angle")],
              ".true."), "after"),
    ("      call mpp_clock_begin (calc_clock)",
     "", "before"),
]
PATCHES["atmos_param/radiation_driver/radiation_driver.F90"].pop()
PATCHES["atmos_param/radiation_driver/radiation_driver.F90"].append(
    ("      call mpp_clock_end (calc_clock)",
     rd_block("rd_calc", [("lw_heatra", "Lw_output(1)%heatra"), ("lw_flxnet", "Lw_output(1)%flxnet"),
                          ("lw_heatracf", "Lw_output(1)%heatracf"), ("lw_flxnetcf", "Lw_output(1)%flxnetcf"),
                          ("sw_dfsw", "Sw_output(1)%dfsw"), ("sw_ufsw", "Sw_output(1)%ufsw"),
                          ("sw_fsw", "Sw_output(1)%fsw"), ("sw_hsw", "Sw_output(1)%hsw"),
                          ("sw_dfswcf", "Sw_output(1)%dfswcf"), ("sw_ufswcf", "Sw_output(1)%ufswcf"),
                          ("sw_hswcf", "Sw_output(1)%hswcf")], "do_rad"), "before"))

SBL_IN = [("t_atm", "ex_t_atm"), ("tr_atm", "ex_tr_atm"), ("u_atm", "ex_u_atm"), ("v_atm", "ex_v_atm"),
          ("p_atm", "ex_p_atm"), ("z_atm", "ex_z_atm"), ("p_surf", "ex_p_surf"), ("t_surf", "ex_t_surf"),
          ("t_ca", "ex_t_ca"), ("tr_surf", "ex_tr_surf"), ("u_surf", "ex_u_surf"), ("v_surf", "ex_v_surf"),
          ("rough_mom", "ex_rough_mom"), ("rough_heat", "ex_rough_heat"), ("rough_moist", "ex_rough_moist"),
          ("rough_scale", "ex_rough_scale"), ("gust", "ex_gust"), ("dt", "dt"), ("land", "ex_land"),
          ("seawater", "ex_seawater"), ("avail", "ex_avail"), ("albedo", "ex_albedo"),
          ("albedo_vis_dir", "ex_albedo_vis_dir"), ("albedo_nir_dir", "ex_albedo_nir_dir"),
          ("albedo_vis_dif", "ex_albedo_vis_dif"), ("albedo_nir_dif", "ex_albedo_nir_dif"),
          ("frac_open_sea", "ex_frac_open_sea"), ("land_frac", "ex_land_frac")]
SF_OUT = [("flux_t", "ex_flux_t"), ("flux_tr", "ex_flux_tr"), ("flux_lw", "ex_flux_lw"), ("flux_u", "ex_flux_u"),
          ("flux_v", "ex_flux_v"), ("cd_m", "ex_cd_m"), ("cd_t", "ex_cd_t"), ("cd_q", "ex_cd_q"),
          ("wind", "ex_wind"), ("u_star", "ex_u_star"), ("b_star", "ex_b_star"), ("q_star", "ex_q_star"),
          ("dhdt_surf", "ex_dhdt_surf"), ("dedt_surf", "ex_dedt_surf"), ("dfdtr_surf", "ex_dfdtr_surf"),
          ("drdt_surf", "ex_drdt_surf"), ("dhdt_atm", "ex_dhdt_atm"), ("dfdtr_atm", "ex_dfdtr_atm"),
          ("dtaudu_atm", "ex_dtaudu_atm"), ("dtaudv_atm", "ex_dtaudv_atm"), ("tr_surf", "ex_tr_surf"),
          ("t_surf", "ex_t_surf"), ("t_ca", "ex_t_ca"), ("rough_mom", "ex_rough_mom"),
          ("rough_heat", "ex_rough_heat"), ("rough_moist", "ex_rough_moist"), ("albedo", "ex_albedo")]
LIAB = [("t", "Land_Ice_Atmos_Boundary%t"), ("albedo", "Land_Ice_Atmos_Boundary%albedo"),
        ("albedo_vis_dir", "Land_Ice_Atmos_Boundary%albedo_vis_dir"),
        ("albedo_nir_dir", "Land_Ice_Atmos_Boundary%albedo_nir_dir"),
        ("albedo_vis_dif", "Land_Ice_Atmos_Boundary%albedo_vis_dif"),
        ("albedo_nir_dif", "Land_Ice_Atmos_Boundary%albedo_nir_dif"),
        ("rough_mom", "Land_Ice_Atmos_Boundary%rough_mom"), ("land_frac", "Land_Ice_Atmos_Boundary%land_frac"),
        ("u_flux", "Land_Ice_Atmos_Boundary%u_flux"), ("v_flux", "Land_Ice_Atmos_Boundary%v_flux"),
        ("dtaudu", "Land_Ice_Atmos_Boundary%dtaudu"), ("dtaudv", "Land_Ice_Atmos_Boundary%dtaudv"),
        ("u_star", "Land_Ice_Atmos_Boundary%u_star"), ("b_star", "Land_Ice_Atmos_Boundary%b_star"),
        ("q_star", "Land_Ice_Atmos_Boundary%q_star"), ("frac_open_sea", "Land_Ice_Atmos_Boundary%frac_open_sea")]
FDOWN = [("dtmass", "ex_dtmass"), ("delta_t", "ex_delta_t"), ("dflux_t", "ex_dflux_t"), ("delta_tr", "ex_delta_tr"),
         ("dflux_tr", "ex_dflux_tr"), ("flux_t", "ex_flux_t"), ("flux_tr", "ex_flux_tr"), ("e_t_n", "ex_e_t_n"),
         ("f_t_delt_n", "ex_f_t_delt_n"), ("e_q_n", "ex_e_q_n"), ("e_tr_n", "ex_e_tr_n"),
         ("f_tr_delt_n", "ex_f_tr_delt_n"), ("dhdt_surf", "ex_dhdt_surf"), ("dedt_surf", "ex_dedt_surf"),
         ("dfdtr_surf", "ex_dfdtr_surf"), ("flux_u", "ex_flux_u"), ("flux_v", "ex_flux_v"),
         ("delta_u", "ex_delta_u"), ("delta_v", "ex_delta_v"), ("flux_lw", "ex_flux_lw"),
         ("flux_lwd", "ex_flux_lwd"), ("flux_sw", "ex_flux_sw"), ("coszen", "ex_coszen")]
FUP = [("t_surf_new", "ex_t_surf_new"), ("dt_t_surf", "ex_dt_t_surf"), ("dt_t_ca", "ex_dt_t_ca"),
       ("delta_t_n", "ex_delta_t_n"), ("delta_tr_n", "ex_delta_tr_n"), ("flux_t", "ex_flux_t"),
       ("flux_tr", "ex_flux_tr"), ("dt_t", "Land_Ice_Atmos_Boundary%dt_t"),
       ("dt_tr", "Land_Ice_Atmos_Boundary%dt_tr"), ("tr_surf_new", "ex_tr_surf_new")]

PATCHES["coupler/flux_exchange.F90"] = [
    ("  use atmos_model_mod,            only: Atm_stock_pe", USE, "after"),
    ("  ex_p_surf = 1\n", "      call ref_dump_set_step(ref_dump_step()+1)", "after"),
    ("  call surface_flux (&", dump_block("sbl_in", SBL_IN), "before"),
    ("       ex_land, ex_seawater .gt. 0,  ex_avail                          )",
     dump_block("sbl_sf", SF_OUT), "after"),
    ("  endif\n#endif\n  zrefm = 10.0", None, None),
    ("  Land_Ice_Atmos_Boundary%t = Land_Ice_Atmos_Boundary%t ** 0.25",
     dump_block("sbl_out", SF_OUT + LIAB), "after"),
    ("  call get_from_xgrid (Land_boundary%t_flux,  'LND', ex_flux_t,    xmap_sfc)",
     dump_block("fdown", FDOWN), "before"),
    ("  call get_from_xgrid (Land_Ice_Atmos_Boundary%dt_t, 'ATM', ex_delta_t_n, xmap_sfc)",
     dump_block("fup", FUP), "after"),
]

PATCHES["atmos_param/moist_processes/moistproc_kernels.F90"] = [
    ("use sat_vapor_pres_mod,         only: compute_qs", USE, "before"),
    ("     if (do_strat .and. (.not.do_liq_num)) then",
     dump_block("ras_in", [("t", "t"), ("q", "q"), ("u", "u"), ("v", "v"), ("pfull", "pfull"),
                           ("phalf", "phalf"), ("zhalf", "zhalf"), ("coldT", "coldT"), ("dt", "dt"),
                           ("tracer", "tracer"), ("tdt", "tdt"), ("qdt", "qdt"), ("udt", "udt"),
                           ("vdt", "vdt"), ("rdt", "rdt"), ("lprec", "lprec"), ("fprec", "fprec")]),
     "before"),
    ("     lprec  = lprec  + rain\n     fprec  = fprec  + snow\n\nend subroutine moistproc_ras",
     None, None),
    ("      call cu_mo_trans (is, js, Time, mc_cmt, t, phalf, pfull, &",
     dump_block("cmt_in", [("mc", "mc_cmt"), ("t", "t"), ("phalf", "phalf"), ("pfull", "pfull"),
                           ("zhalf", "zhalf"), ("zfull", "zfull"), ("dt", "dt"), ("u", "u"), ("v", "v"),
                           ("tracer", "tracer"), ("pmass", "pmass"), ("det", "det_cmt")]), "before"),
    ("                        qtr, diff_cu_mo  )",
     dump_block("cmt_out", [("utnd", "utnd"), ("vtnd", "vtnd"), ("ttnd", "ttnd"), ("qtr", "qtr"),
                            ("diff_cu_mo", "diff_cu_mo")]), "after"),
    ("        call strat_cloud (Time, is, ie, js, je, dt, pfull, phalf,     & \n                          radturbten, t, q, tracer(:,:,:,nql),",
     dump_block("sc_in", [("dt", "dt"), ("pfull", "pfull"), ("phalf", "phalf"), ("radturbten", "radturbten"),
                          ("t", "t"), ("q", "q"), ("ql", "tracer(:,:,:,nql)"), ("qi", "tracer(:,:,:,nqi)"),
                          ("qa", "tracer(:,:,:,nqa)"), ("omega", "omega"), ("mc_full", "mc_full"),
                          ("diff_t", "diff_t"), ("land", "land"),
                          ("chr", "convective_humidity_ratio"), ("cha", "convective_humidity_area")]),
     "before"),
    ("                          limit_conv_cloud_frac, mask=mask)\n      endif",
     dump_block("sc_out", [("ttnd", "ttnd"), ("qtnd", "qtnd"), ("ql_tnd", "q_tnd(:,:,:,nql)"),
                           ("qi_tnd", "q_tnd(:,:,:,nqi)"), ("qa_tnd", "q_tnd(:,:,:,nqa)"),
                           ("f_snow_berg", "f_snow_berg"), ("rain3d", "rain3d"), ("snow3d", "snow3d"),
                           ("snowclr3d", "snowclr3d"), ("rain", "rain"), ("snow", "snow")]), "after"),
]
PATCHES["atmos_param/moist_processes/moistproc_kernels.F90"][2] = (
    "     lprec  = lprec  + rain\n     fprec  = fprec  + snow\n\nend subroutine moistproc_ras",
    dump_block("ras_out", [("t", "t"), ("q", "q"), ("u", "u"), ("v", "v"), ("tracer", "tracer"),
                           ("ttnd", "ttnd"), ("qtnd", "qtnd"), ("mc", "mc"), ("det0", "det0"),
                           ("rain", "rain"), ("snow", "snow"), ("q_tnd", "q_tnd"), ("tdt", "tdt"),
                           ("qdt", "qdt"), ("udt", "udt"), ("vdt", "vdt"), ("rdt", "rdt"),
                           ("lprec", "lprec"), ("fprec", "fprec")]), "before_last")

PATCHES["atmos_param/physics_driver/physics_driver.F90"] += [
    ("                           radturbten(is:ie,js:je,:),                &\n                           cush           (is:ie,js:je),          &!",
     dump_block("moist_in", [("dt", "dt"), ("frac_land", "frac_land"), ("omega", "omega"),
                             ("diff_t", "diff_t(is:ie,js:je,:)"), ("radturbten", "radturbten(is:ie,js:je,:)"),
                             ("u_star", "u_star"), ("b_star", "b_star"), ("q_star", "q_star"),
                             ("pbltop", "pbltop(is:ie,js:je)"), ("area", "area"), ("lat", "lat")]
                + GRID + CUR + PREV + TEND), "before_first_of_call"),
    ("        diff_cu_mo(is:ie, js:je,:) = diff_cu_mo_loc(:,:,:)",
     dump_block("moist_out", TEND + [("diff_cu_mo", "diff_cu_mo_loc"), ("convect", "convect(is:ie,js:je)"),
                                     ("lprec", "lprec"), ("fprec", "fprec"), ("gust_cv", "gust_cv"),
                                     ("lsc_cloud_area", "lsc_cloud_area(is:ie,js:je,:)"),
                                     ("lsc_liquid", "lsc_liquid(is:ie,js:je,:)"),
                                     ("lsc_ice", "lsc_ice(is:ie,js:je,:)")]), "before"),
]

# --- longwave (sealw99) intermediates ---------------------------------------------------
OPT = ["empl1f", "empl2f", "vrpfh2o", "xch2obd", "tphfh2o", "avephif", "totaerooptdep", "empl1", "empl2",
       "var1", "var2", "emx1f", "emx2f", "totvo2", "avephi", "totch2obdwd", "xch2obdwd", "totphi", "cntval",
       "toto3", "tphio3", "var3", "var4", "wk", "rh2os", "rfrgn", "tfac", "totaerooptdep_15", "totf11",
       "totf12", "totf113", "totf22", "emx1", "emx2", "csfah2o", "aerooptdep_KE_15"]
GTF = ["a1", "a2", "tdav", "tlsqu", "tmpdiff", "tstdav", "co2nbl", "n2o9c", "tn2o17", "co2spnb"]
LWD = ["flx1e1", "gxcts", "flx1e1f", "excts", "fctsg", "fluxn", "fluxncf", "exctsn", "cts_out", "cts_outcf"]
TABS = ([(f"{t}_{c}", f"{t}%{c}") for t in ("tab1", "tab2", "tab3", "tab1w", "tab1a", "tab2a", "tab3a")
         for c in ("vae", "td", "md", "cd")] + [(f"tabsr_{c}", f"tabsr%{c}") for c in ("vae", "td")])
LWDIAG = [(f"d_{n}", f"Lw_diagnostics%{n}") for n in LWD]
PATCHES["atmos_param/sea_esf_rad/sealw99.F90"] = [
    ("use mpp_mod,             only: input_nml_file", USE, "before"),
    ("      call co2_source_calc (Atmos_input, Rad_gases, sorc, Gas_tf, &\n                            source_band, dsrcdp_band)",
     rd_block("lw_a", [(f"op_{n}", f"Optical%{n}") for n in OPT] + [(f"gt_{n}", f"Gas_tf%{n}") for n in GTF]
              + [("sorc", "sorc"), ("source_band", "source_band"), ("dsrcdp_band", "dsrcdp_band"),
                 ("apcm", "apcm"), ("bpcm", "bpcm"), ("atpcm", "atpcm"), ("btpcm", "btpcm"), ("acomb", "acomb"),
                 ("bcomb", "bcomb"), ("cld_indx", "cld_indx")] + TABS, ".true."), "after"),
    ("      call cloud (KS, Cldrad_props, Cld_spec, Lw_clouds, cldtf)",
     rd_block("lw_b", [("cldtf", "cldtf"), ("co21c", "co21c"), ("co21r", "co21r"), ("tch4n2oe", "tch4n2oe"),
                       ("to3cnt", "to3cnt"), ("overod", "overod"), ("cnttaub1", "cnttaub1"),
                       ("cnttaub2", "cnttaub2"), ("cnttaub3", "cnttaub3"),
                       ("taucld_rndlw", "Lw_clouds%taucld_rndlw"), ("taucld_mxolw", "Lw_clouds%taucld_mxolw")],
              ".true."), "after"),
    ("                                 cts_sumcf, gxctscf, including_aerosols)",
     rd_block("lw_c", [("cts_sum", "cts_sum"), ("cts_sumcf", "cts_sumcf"), ("gxctscf", "gxctscf")] + LWDIAG,
              ".true."), "after"),
    ("                    cld_indx, flx1e1cf,  tcfc8, including_aerosols)",
     rd_block("lw_d", [("e1ctw1", "e1ctw1"), ("e1ctw2", "e1ctw2"), ("trans_band1", "trans_band1"),
                       ("trans_band2", "trans_band2"), ("flx1e1cf", "flx1e1cf"), ("tcfc8", "tcfc8"),
                       ("tch4n2oe", "tch4n2oe")] + LWDIAG, ".true."), "after"),
    ("    call longwave_fluxes_ks (source_band, trans_band1, dsrcdp_band,  &\n                             trans_band2, cldtf, cld_indx , &\n                                          Lw_diagnostics)",
     rd_block("lw_e", [("trans_band1", "trans_band1"), ("trans_band2", "trans_band2")] + LWDIAG, ".true."),
     "after"),
    ("                                Lw_diagnostics, e1ctw1)",
     rd_block("lw_f", LWDIAG, ".true."), "after"),
    ("                                  trans_b2d2, cldtf, cld_indx,  &\n                                  Lw_diagnostics )",
     rd_block("lw_g", [("trans_b2d1", "trans_b2d1"), ("trans_b2d2", "trans_b2d2"), ("emspec", "emspec"),
                       ("emspecf", "emspecf"), ("co21c_KEp1", "co21c_KEp1"), ("co21r_KEp1", "co21r_KEp1"),
                       ("cldtf", "cldtf"), ("to3cnt", "to3cnt"), ("overod", "overod"), ("tcfc8", "tcfc8"),
                       ("tch4n2oe", "tch4n2oe")] + LWDIAG, ".true."), "after"),
    ("                               cld_indx, Lw_diagnostics )",
     rd_block("lw_h", [("trans_band1", "trans_band1"), ("emisdg", "emisdg"), ("emisdgf", "emisdgf"),
                       ("contdg", "contdg"), ("to3dg", "to3dg"), ("co21c", "co21c"), ("cldtf", "cldtf")] + LWDIAG,
              ".true."), "after"),
    ("      call lw_clouds_dealloc (Lw_clouds)",
     rd_block("lw_i", [("heatra", "Lw_output%heatra"), ("flxnet", "Lw_output%flxnet"),
                       ("heatracf", "Lw_output%heatracf"), ("flxnetcf", "Lw_output%flxnetcf"),
                       ("bdy_flx", "Lw_output%bdy_flx"), ("bdy_flx_clr", "Lw_output%bdy_flx_clr"),
                       ("cts_sum", "cts_sum"), ("cts_sumcf", "cts_sumcf"), ("flx", "flx"), ("flxcf", "flxcf"),
                       ("heatem", "heatem"), ("heatemcf", "heatemcf")] + LWDIAG, ".true."), "before"),
]
GT_TABS = ["co251", "co258", "cdt51", "cdt58", "c2d51", "c2d58", "co2m51", "co2m58", "cdtm51", "cdtm58",
           "c2dm51", "c2dm58", "co211", "co218", "co215nbps1", "co215nbps8", "co2dt15nbps1", "co2dt15nbps8",
           "co2d2t15nbps1", "co2d2t15nbps8", "ch451", "ch458", "n2o51", "n2o58", "n2o71", "n2o78", "n2o91",
           "n2o98", "stemp", "gtemp"]
PATCHES["atmos_param/sea_esf_rad/gas_tf.F90"] = [
    ("use mpp_mod,             only : input_nml_file", USE, "before"),
    ("      press(:,:,:) = 10.0*Atmos_input%press(:,:,:)\n      pflux(:,:,:) = 10.0*Atmos_input%pflux(:,:,:)\n      tflux(:,:,:)",
     "      if (allocated(co251)) then\n        call ref_dump_begin('gt_tab')\n"
     + "".join(f"        if (allocated({n})) call ref_dump('{n}', {n})\n" for n in GT_TABS)
     + "        call ref_dump('ixprkminh2o', ixprkminh2o)\n        call ref_dump_end()\n      endif", "before"),
]


# local variables needed by the hooks in atmosphere_down
DECLS = {
    "atmos_scm/driver/coupled/atmosphere.F90": [
        ("  type(time_type) :: Time_prev, Time_next\n!-----------------------------------------------------------------------\n!---- dynamics -----",
         "  integer :: rd_sec, rd_day"),
    ],
}


def apply(path, patches):
    src = os.path.join(REF, "overrides", path)
    if not os.path.exists(src):
        src = os.path.join(REF, "src", path)
    text = open(src).read()
    for anchor, ins in DECLS.get(path, []):
        n = text.count(anchor)
        if n != 1:
            sys.exit(f"{path}: decl anchor found {n} times: {anchor[:60]!r}")
        i = text.index(anchor)
        j = text.index("\n", i)          # after the first line of the anchor
        text = text[:j + 1] + ins + "\n" + text[j + 1:]
    for anchor, ins, where in patches:
        if ins is None:
            continue
        n = text.count(anchor)
        if n != 1:
            sys.exit(f"{path}: anchor found {n} times: {anchor[:60]!r}")
        i = text.index(anchor)
        if where == "before_first_of_call":
            # before the statement that contains the anchor (walk back over continuation lines)
            j = text.rfind("\n", 0, i) + 1
            while True:
                pj = text.rfind("\n", 0, j - 1) + 1
                prev = text[pj:j - 1]
                if prev.rstrip().endswith("&") or prev.split("!")[0].rstrip().endswith("&"):
                    j = pj
                else:
                    break
            text = text[:j] + ins + "\n" + text[j:]
        elif where == "before_last":
            j = text.rfind("\n", 0, i + len(anchor)) + 1
            text = text[:j] + ins + "\n" + text[j:]
        elif where == "before":
            j = text.rfind("\n", 0, i) + 1
            text = text[:j] + ins + "\n" + text[j:]
        else:
            # after the line holding the end of the anchor (anchors may span lines)
            j = text.index("\n", i + len(anchor) - 1)
            text = text[:j + 1] + ins + "\n" + text[j + 1:]
    out = os.path.join(REF, "hooks", path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(text)
    print("hooked", path)


if __name__ == "__main__":
    for p, ps in PATCHES.items():
        apply(p, ps)
