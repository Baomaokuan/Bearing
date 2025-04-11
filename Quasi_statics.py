# -*- coding: utf-8 -*-
"""
Created on Tue Feb 01 8:00:00 2025

@author: Baomaokuan's Chengguo
Program: Bearing Analysis of Mechanical Kinetics-b(V1.0a) Quasi_statics
"""

###############################################################################
#                                Input library                                #
###############################################################################
import os
import math
import pickle
import numpy as np
import scipy.optimize as opt

from Input import *
from Mat_properties import *
from Oil_properties import *

from scipy import special
from numba import vectorize

__all__ = [
    "ball_centrifugal_forece",
    "no_load_position",
    "initial_position",
    "quasi_static_for_ball_bearing"
]
###############################################################################
#                               Tool function 0                               #
###############################################################################
# @njit(fastmath=False)
def value_1(x, y):
    xs0, xs1, xs2 = x.shape[0], x.shape[1], x.shape[2]
    ys0, ys1, ys2 = y.shape[0], y.shape[1], y.shape[2]
    if xs0 < ys0:
        xx = np.zeros((ys0, xs1, xs2))
        xx[:, :, 0::] = x[:, :, :]
        x = xx
    elif xs0 > ys0:
        yy = np.zeros((xs0, ys1, ys2))
        yy[:, :, 0::] = y[:, :, :]
        y = yy
    if xs2 < ys2:
        xx = np.zeros_like(y)
        xx[:, :, 0::] = x[:, :, :]
        x = xx
    elif xs2 > ys2:
        yy = np.zeros_like(x)
        yy[:, :, 0::] = y[:, :, :]
        y = yy
    z = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[2]):
            z[i, 0, j] = x[i, 1, j] * y[i, 2, j] - x[i, 2, j] * y[i, 1, j]
            z[i, 1, j] = x[i, 2, j] * y[i, 0, j] - x[i, 0, j] * y[i, 2, j]
            z[i, 2, j] = x[i, 0, j] * y[i, 1, j] - x[i, 1, j] * y[i, 0, j]

    return z

###############################################################################
#                              Tool function 1                                #
###############################################################################
# @njit(fastmath=False)
def value_2(a, b, tj):
    k0, k1, k2 = (
        0.5 * (a - 1) + 0.5 * (a + 1) * tj,
        0.5 * (a + b) + 0.5 * (b - a) * tj,
        0.5 * (1 + b) + 0.5 * (1 - b) * tj
    )

    return k0, k1, k2

###############################################################################
#                              Tool function 2                                #
###############################################################################
# @njit(fastmath=False)
def value_4(a, b):
    as1, as2, bs0 = a.shape[1], a.shape[2], b.shape[0]
    aa = np.zeros((bs0, as1, as2))
    num = -1
    for i in range(bs0):
        if b[i, 0, 0] != 0:
            num = num + 1
            for j in range(as1):
                for k in range(as2):
                    aa[i, j, k] = a[num, j, k]

    return aa

###############################################################################
#                   Calculate the flatdisk clearance change                   #
###############################################################################
def flatdisk(den, E, po, r0, r1, r):
    """Solve the clearance change.

    Parameters
    ----------
    den: float
        Density.
    E: float
        Elastic modulus.
    po: float
        Poissons ratio.
    r0: float
        Outer radius.
    r1: float
        Inner radius.
    r: float
        Radius at which the solutions are required.

    Returns
    -------
    u0: float
        Displacement under unit external pressure.
    u1: float
        Displacement under unit internal pressure.
    u2: float
        Displacement due to unit rotational velocity.
    st0: float
        Stress under unit external pressure.
    st1: float
        Stress under unit internal pressure.
    st2: float
        Stress due to unit rotational velocity.
    """
    if r1 > 0:
        c0 = (r0 / r) ** 2
        c1 = (r1 / r) ** 2
        c2 = r0 ** 2 - r1 ** 2

        u0 = -r * ((1 - po) + (1 + po) * c1) * r0 ** 2 / (E * c2)
        u1 = r * ((1 - po) + (1 + po) * c0) * r1 ** 2 / (E * c2)
        u2 = den * r * 0.125 * (3 + po) * (1 - po) * (
            r0 ** 2 + r1 ** 2 +
            (1 + po) * (r0 * r1 / r) ** 2 / (1 - po) -
            (1 + po) * r ** 2 / (3 + po)
        ) / E

        st0 = -(1 + c1) * r0 ** 2 / c2
        st1 = (1 + c0) * r1 ** 2 / c2
        st2 = den * 0.125 * (3 + po) * (
            r0 ** 2 + r1 ** 2 +
            (r0 * r1 / r) ** 2 -
            (1 + 3 * po) * r ** 2 / (3 + po)
        )
    else:
        u0 = -r * (1 - po) / E
        u1 = u2 = st1 = 0.
        st0 = -1.
        st2 = den * 0.125 * (3 + po) * (
            r0 ** 2 - (1 + 3 * po) * r ** 2 / (3 + po)
        )

    return u0, u1, u2, st0, st1, st2

###############################################################################
#                        Calculate the clearance change                       #
###############################################################################
def expansion_constant(denh, eh, poh, dh, dio, dens, es, pos, dii, ds, deno,
                       eo, poo, dyo, deni, ei, poi, dyi, denc, ec, poc, dco,
                       dci, cteh, ctes, cteo, ctei, ctec, nc):
    """Solve the clearance change.

    Parameters
    ----------
    denh: float
        Density of the housing.
    eh: float
        Elastic modulus of the housing.
    poh: float
        Poissons ratio of the housing.
    dh: float
        Inner radius of the housing.
    dio: float
        Outer radius of the outer race.
    dens: float
        Density of the shaft.
    es: float
        Elastic modulus of the housing.
    pos: float
        Poissons ratio of the housing.
    dii: float
        Inner radius of the inner race.
    ds: float
        Outer radius of the shaft.
    deno: float
        Density of the outer race.
    eo: float
        Elastic modulus of the outer race.
    poo: float
        Poissons ratio of the outer race.
    dyo: float
        Inner radius of the outer race.
    deni: float
        Density of the inner race.
    ei: float
        Elastic modulus of the inner race.
    poi: float
        Poissons ratio of the inner race.
    dyi: float
        Outer radius of the inner race.
    denc: float
        Density of the cage.
    ec: float
        Elastic modulus of the cage.
    poc: float
        Poissons ratio of the cage.
    dco: float
        Outer radius of the cage.
    dci: float
        Inner radius of the cage.
    cteh: float
        Coefficient of housing thermal expansion.
    ctes float
        Coefficient of shaft thermal expansion.
    cteo: float
        Coefficient of outer race thermal expansion.
    ctei: float
        Coefficient of inner race thermal expansion.
    ctec: float
        Coefficient of cage thermal expansion.
    nc: float
        Number of cage segments in the Bearing.

    Returns
    -------
    Info_e: tuple
        Information of expansion.
    """
    deltah0, deltah1, deltah2, hooph0, hooph1, hooph2 = flatdisk(
        denh, eh, poh, 0.5 * dh, 0.5 * dio, 0.5 * dio
    )
    vh00, vh01, vh02 = deltah0, deltah1, deltah2
    sth00, sth01, sth02 = hooph0, hooph1, hooph2
    vh03 = 0.5 * dio * cteh

    deltas0, deltas1, deltas2, hoops0, hoops1, hoops2 = flatdisk(
        dens, es, pos, 0.5 * dii, 0.5 * ds, 0.5 * dii
    )
    vs00, vs01, vs02 = deltas0, deltas1, deltas2
    sts00, sts01, sts02 = hoops0, hoops1, hoops2
    vs03 = 0.5 * dii * ctes

    deltao10, deltao11, deltao12, hoopo10, hoopo11, hoopo12 = flatdisk(
        deno, eo, poo, 0.5 * dio, 0.5 * dyo, 0.5 * dio
    )
    vo00, vo01, vo02 = deltao10, deltao11, deltao12
    sto00, sto01, sto02 = hoopo10, hoopo11, hoopo12
    vo03 = 0.5 * dio * cteo

    deltao20, deltao21, deltao22, hoopo20, hoopo21, hoopo22 = flatdisk(
        deno, eo, poo, 0.5 * dio, 0.5 * dyo, 0.5 * dyo
    )
    vo10, vo11, vo12 = deltao20, deltao21, deltao22
    sto10, sto11, sto12 = hoopo20, hoopo21, hoopo22
    vo13 = 0.5 * dyo * cteo

    deltai10, deltai11, deltai12, hoopi10, hoopi11, hoopi12 = flatdisk(
        deni, ei, poi, 0.5 * dyi, 0.5 * dii, 0.5 * dyi
    )
    vi00, vi01, vi02 = deltai10, deltai11, deltai12
    sti00, sti01, sti02 = hoopi10, hoopi11, hoopi12
    vi03 = 0.5 * dyi * ctei

    deltai20, deltai21, deltai22, hoopi20, hoopi21, hoopi22 = flatdisk(
        deni, ei, poi, 0.5 * dyi, 0.5 * dii, 0.5 * dii
    )
    vi10, vi11, vi12 = deltai20, deltai21, deltai22
    sti10, sti11, sti12 = hoopi20, hoopi21, hoopi22
    vi13 = 0.5 * dii * ctei

    if nc <= 0:
        vc00, vc01, vc02, vc03 = 0., 0., 0., 0.
        vc10, vc11, vc12, vc13 = 0., 0., 0., 0.
        stc00, stc01, stc02 = 0., 0., 0.
        stc10, stc11, stc12 = 0., 0., 0.
    else:
        deltac10, deltac11, deltac12, hoopc10, hoopc11, hoopc12 = flatdisk(
            denc, ec, poc, 0.5 * dco, 0.5 * dci, 0.5 * dco
        )
        vc00, vc01, vc02 = deltac10, deltac11, deltac12
        stc00, stc01, stc02 = hoopc10, hoopc11, hoopc12
        vc03 = 0.5 * dco * ctec

        deltac20, deltac21, deltac22, hoopc20, hoopc21, hoopc22 = flatdisk(
            denc, ec, poc, 0.5 * dco, 0.5 * dci, 0.5 * dci
        )
        vc10, vc11, vc12 = deltac20, deltac21, deltac22
        stc10, stc11, stc12 = hoopc20, hoopc21, hoopc22
        vc13 = 0.5 * dci * ctec
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    Info_ec = (vh00,
               # Housing coefficients for expansion.
               vh01,
               # Housing coefficients for expansion.
               vh02,
               # Housing coefficients for expansion.
               vh03,
               # Housing coefficients for expansion.
               vs00,
               # Shaft coefficients for expansion.
               vs01,
               # Shaft coefficients for expansion.
               vs02,
               # Shaft coefficients for expansion.
               vs03,
               # Shaft coefficients for expansion.
               vo00,
               # Outer race coefficients for expansion.
               vo01,
               # Outer race coefficients for expansion.
               vo02,
               # Outer race coefficients for expansion.
               vo03,
               # Outer race coefficients for expansion.
               vo10,
               # Outer race coefficients for expansion.
               vo11,
               # Outer race coefficients for expansion.
               vo12,
               # Outer race coefficients for expansion.
               vo13,
               # Outer race coefficients for expansion.
               vi00,
               # Inner race coefficients for expansion.
               vi01,
               # Inner race coefficients for expansion.
               vi02,
               # Inner race coefficients for expansion.
               vi03,
               # Inner race coefficients for expansion.
               vi10,
               # Inner race coefficients for expansion.
               vi11,
               # Inner race coefficients for expansion.
               vi12,
               # Inner race coefficients for expansion.
               vi13,
               # Inner race coefficients for expansion.
               vc00,
               # Cage coefficients for expansion.
               vc01,
               # Cage coefficients for expansion.
               vc02,
               # Cage coefficients for expansion.
               vc03,
               # Cage coefficients for expansion.
               vc10,
               # Cage coefficients for expansion.
               vc11,
               # Cage coefficients for expansion.
               vc12,
               # Cage coefficients for expansion.
               vc13,
               # Cage coefficients for expansion.
               sth00,
               # Housing coefficients for hoop stress.
               sth01,
               # Housing coefficients for hoop stress.
               sth02,
               # Housing coefficients for hoop stress.
               sts00,
               # Shaft coefficients for hoop stress.
               sts01,
               # Shaft coefficients for hoop stress.
               sts02,
               # Shaft coefficients for hoop stress.
               sto00,
               # Outer race coefficients for hoop stress.
               sto01,
               # Outer race coefficients for hoop stress.
               sto02,
               # Outer race coefficients for hoop stress.
               sto10,
               # Outer race coefficients for hoop stress.
               sto11,
               # Outer race coefficients for hoop stress.
               sto12,
               # Outer race coefficients for hoop stress.
               sti00,
               # Inner race coefficients for hoop stress.
               sti01,
               # Inner race coefficients for hoop stress.
               sti02,
               # Inner race coefficients for hoop stress.
               sti10,
               # Inner race coefficients for hoop stress.
               sti11,
               # Inner race coefficients for hoop stress.
               sti12,
               # Inner race coefficients for hoop stress.
               stc00,
               # Cage coefficients for hoop stress.
               stc01,
               # Cage coefficients for hoop stress.
               stc02,
               # Cage coefficients for hoop stress.
               stc10,
               # Cage coefficients for hoop stress.
               stc11,
               # Cage coefficients for hoop stress.
               stc12
               # Cage coefficients for hoop stress.
               )

    return Info_ec

###############################################################################
#                         Calculate expansion subcall                         #
###############################################################################
# @njit(fastmath=False)
def expansion_subcall(x, Info_tc, mod_es):
    """Solve the clearance change.

    Parameters
    ----------
    x: np.darray
        Solution vector.
    Info_tc: tuple
        Information of the temperature change.
    mod_es: tuple
        Mode data of expansion_subcall.

    Returns
    -------
    Info_es: tuple
        Information of expansion_subcall.
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    (D_p_u,
     R_o_m,
     Fit_i_s,
     Fit_o_h,
     n,
     n_cseg,
     vh01,
     vh02,
     vh03,
     vs00,
     vs02,
     vs03,
     vo00,
     vo02,
     vo03,
     vo10,
     vo12,
     vo13,
     vi01,
     vi02,
     vi03,
     vi11,
     vi12,
     vi13,
     vc02,
     vc03,
     vc12,
     vc13,
     sto10,
     sto12,
     sti01,
     sti02,
     sti11,
     sti12,
     stc12
     ) = mod_es[0::]
    ###########################################################################
    #                               End prepare                               #
    ###########################################################################
    temp_o, temp_i, temp_h, temp_s, temp_c, temp_r = Info_tc[0:6]

    fitso, fitsi = 0.5 * Fit_o_h, 0.5 * Fit_i_s

    race_ang_vel_o = (
        (math.cos(x[32+12*n]) * math.cos(x[34+12*n]) * x[31+12*n] +
         math.sin(x[34+12*n]) * x[33+12*n])
    )
    race_ang_vel_i = (
        math.cos(x[8]) * math.cos(x[10]) * x[7] + math.sin(x[10]) * x[9]
    )
    cage_ang_vel = (
        math.cos(x[20]) * math.cos(x[22]) * x[19] + math.sin(x[22]) * x[21]
    )
    ###########################################################################
    #                             Geometry change                             #
    ###########################################################################
    race_hoop_o = sto12 * race_ang_vel_o ** 2
    race_fhoop_o = race_hoop_o

    race_hoop_i = sti12 * race_ang_vel_i ** 2
    race_fhoop_i = sti02 * race_ang_vel_i ** 2

    race_u_o_0 = vo02 * race_ang_vel_o ** 2 + vo03 * (temp_o - temp_r)
    race_u_o_1 = vo12 * race_ang_vel_o ** 2 + vo13 * (temp_o - temp_r)

    race_u_i_0 = vi02 * race_ang_vel_i ** 2 + vi03 * (temp_i - temp_r)
    race_u_i_1 = vi12 * race_ang_vel_i ** 2 + vi13 * (temp_i - temp_r)

    hsng_u = vh02 * race_ang_vel_o ** 2 + vh03 * (temp_h - temp_r)
    op_race_fit_o = fitso + race_u_o_0 - hsng_u
    if op_race_fit_o > 0:
        P_o = op_race_fit_o / (vh01 - vo00)
        race_u_o_0 = race_u_o_0 + P_o * vo00
        race_u_o_1 = race_u_o_1 + P_o * vo10
        race_hoop_o = race_hoop_o + P_o * sto10
        race_fhoop_o = race_fhoop_o + race_hoop_o

    shft_u = vs02 * race_ang_vel_i ** 2 + vs03 * (temp_s - temp_r)
    op_race_fit_i = fitsi + shft_u - race_u_i_1
    if op_race_fit_i > 0:
        P_i = op_race_fit_i / (vi11 - vs00)
        race_u_i_0 = race_u_i_0 + P_i * vi01
        race_u_i_1 = race_u_i_1 + P_i * vi11
        race_hoop_i = race_hoop_i + P_i * sti11
        race_fhoop_i = race_fhoop_i + P_i * sti01

    op_race_fit_o, op_race_fit_i = 2 * op_race_fit_o, 2 * op_race_fit_i
    race_exp_o, race_exp_i = race_u_o_1, race_u_i_0

    if n_cseg <= 0:
        cage_u_0, cage_u_1, cage_poc_u = 0., 0., 0.
    else:    
        cage_u_0 = vc02 * cage_ang_vel ** 2 + vc03 * (temp_c - temp_r)
        cage_u_1 = vc12 * cage_ang_vel ** 2 + vc13 * (temp_c - temp_r)
        cage_poc_u = cage_u_0 * D_p_u / R_o_m
        cage_hoop = stc12 * cage_ang_vel ** 2

    tauh_o_GZ = -0.5 * race_fhoop_o
    tauh_i_GZ = -0.5 * race_fhoop_i
    tauh_o_IH = -0.4714045 * race_fhoop_o
    tauh_i_IH = -0.4714045 * race_fhoop_i
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    Info_es = (race_u_o_0,
               # Expansion at outer radius of outer race.
               race_u_o_1,
               # Expansion at inner radius of outer race.
               race_u_i_0,
               # Expansion at outer radius of inner race.
               race_u_i_1,
               # Expansion at inner radius of inner race.
               hsng_u,
               # Housing expansion.
               shft_u,
               # Shaft expansion.
               cage_u_0,
               # Expansion at outer radius of cage.
               cage_u_1,
               # Expansion at outer radius of cage.
               cage_poc_u,
               # Cage pocket expansion.
               race_exp_o,
               # Race expansion for ball/outer race interaction.
               race_exp_i,
               # Race expansion for ball/inner race interaction.
               op_race_fit_o,
               # Operating fits as diametral of outer race.
               op_race_fit_i,
               # Operating fits as diametral of inner race.
               race_hoop_o,
               # Rotational hoop at inner radius of outer race.
               race_hoop_i,
               # Rotational hoop at inner radius of inner race.
               cage_hoop,
               # Rotational hoop at inner radius of cage.
               race_fhoop_o,
               # Outer race hoop stress for fatigue life adjustment.
               race_fhoop_i,
               # Inner race hoop stress for fatigue life adjustment.
               tauh_o_GZ,
               # Max shear stress due to hoop stress in outer race, GZ model.
               tauh_i_GZ,
               # Max shear stress due to hoop stress in inner race, GZ model.
               tauh_o_IH,
               # Max shear stress due to hoop stress in outer race, IH model.
               tauh_i_IH
               # Max shear stress due to hoop stress in inner race, IH model.
               )

    return Info_es

###############################################################################
#                          Calculate hertz stiffness                          #
###############################################################################
# @njit(fastmath=False)
def hertz_stiffness(r0x, r0y, r1x, r1y, E0, po0, E1, po1, stype):
    """Solve the hertz stiffness.

    Parameters
    ----------
    r0x: float
        Equivalent radius of deformed surface in x direction of part I.
    r0y: float
        Equivalent radius of deformed surface in y direction of part II.
    r1x: float
        Equivalent radius of deformed surface in x direction of part I.
    r1y: float
        Equivalent radius of deformed surface in y direction of part II.
    E0: float
        Elastic Modulus of part I.
    po0: float
        Poison number of part I.
    E1: float
        Elastic Modulus of part II.
    po1: float
        Poison number of part II.
    stype: float
        Judgement of special conditions.

    Returns
    -------
    K: float
        Coefficient of load-deformerd.
    k: float
        Value of a/b ratio.
    e1: float
        Elliptic integrals.
    e2: float
        Elliptic integrals.
    E: float
        Equivalent elastic Modulus.
    e: float
        Coefficient of equivalent elastic Modulus.
    rx: float
        Equivalent radius in x direction.
    r: float
        Equivalent radius.
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    fri = np.array([
        0.000, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450,
        0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.810, 0.820, 0.830,
        0.840, 0.850, 0.860, 0.870, 0.880, 0.890, 0.900, 0.910, 0.920, 0.930,
        0.940, 0.945, 0.950, 0.955, 0.960, 0.965, 0.970, 0.975, 0.980, 0.985,
        0.990, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999,
        0.9999
        ])
    bai = np.array([
        1.0000000000, 0.9354578300, 0.8748038700, 0.8175561800, 0.7632987900,
        0.7116682500, 0.6623427500, 0.6150329800, 0.5694741000, 0.5254183800,
        0.4826277800, 0.4408657100, 0.3998870200, 0.3594240800, 0.3191656700,
        0.2787208200, 0.2375498400, 0.2291628700, 0.2207063700, 0.2121703400,
        0.2035434100, 0.1948125100, 0.1859624700, 0.1769754900, 0.1678304500,
        0.1585018600, 0.1489585000, 0.1391613800, 0.1290605900, 0.1185904900,
        0.1076614800, 0.1019870200, 0.0961455750, 0.0901105670, 0.0838479360,
        0.0773128060, 0.0704439150, 0.0631536790, 0.0553087850, 0.0466871050,
        0.0368613040, 0.0346808840, 0.0324019250, 0.0300058470, 0.0274672900,
        0.0247500000, 0.0217987960, 0.0185217170, 0.0147417790, 0.0100106380,
        0.0028273958
        ])
    ae = np.array([
        0.57079633000, 0.52052763000, 0.47410951000, 0.43110337000,
        0.39113752000, 0.35389475000, 0.31910272000, 0.28652637000,
        0.25596181000, 0.22723154000, 0.20018053000, 0.17467313000,
        0.15059059000, 0.12782919000, 0.10629891000, 0.08592276400,
        0.06663701600, 0.06290613200, 0.05921655300, 0.05556800900,
        0.05196026800, 0.04839313700, 0.04486647600, 0.04138019600,
        0.03793427500, 0.03452876800, 0.03116382300, 0.02783970500,
        0.02455682900, 0.02131580400, 0.01811751300, 0.01653476900,
        0.01496322500, 0.01340313300, 0.01185480600, 0.01031863400,
        0.00879511870, 0.00728491930, 0.00578894050, 0.00430849980,
        0.00284570990, 0.00255556420, 0.00226631770, 0.00197802860,
        0.00169076910, 0.00140463280, 0.00111974580, 0.00083628995,
        0.00055455462, 0.00027509663, 0.00002698209
        ])
    ###########################################################################
    #                                Stiffness                                #
    ###########################################################################
    E = 2 / (((1 - po0 ** 2) / E0) + ((1 - po1 ** 2) / E1))
    e = 2 / E
    rx = r0x * r1x / (r0x + r1x)
    ry = r0y * r1y / (r0y + r1y)
    r = rx * ry / (rx + ry)
    sum_ = 1 / r0x + 1 / r0y + 1 / r1x + 1 / r1y

    if stype == 1:
        fr = abs((1 / r0x + 1 / r1x - 1 / r0y - 1 / r1y) / sum_)
    else:
        fr = (1 / r0x + 1 / r1x - 1 / r0y - 1 / r1y) / sum_

    if fr <= 0:
        a_ = 1.
        b_ = 1.
        del_ = 1.
        e1 = 0.
        e2 = 0.
        k = 1.
        K = 2 * ((2 / del_) ** 3 / sum_) ** 0.5 / 1.5 * E
    elif fr >= 1:
        a_= 10.
        b_ = 0.
        del_ = 0.
        e1 = 0.
        e2 = 0.
        k = 1e10
        K = 2 * ((2 / del_) ** 3 / sum_) ** 0.5 / 1.5 * E
    else:
        fris0 = fri.shape[0]
        if fr <= fri[0]:
            m0, m1 = 0, 1
        else:
            i = 0
            while fr > fri[i] and i < fris0 - 1:
                i = i + 1
            m0, m1 = i - 1, i
        fr_ = (fr - fri[m0]) / (fri[m1] - fri[m0])
        ba = bai[m0] + (bai[m1] - bai[m0]) * fr_
        e1 = ae[m0] + (ae[m1] - ae[m0]) * fr_ + 1
        e2 = (1 + ba ** 2 - fr * (1 - ba ** 2)) * e1 / (2 * ba ** 2)
        del_ = (2 * e2 / math.pi) * (math.pi * ba ** 2 / (2 * e1)) ** (1 / 3)
        a_ = (2 * e1 / (math.pi * ba ** 2)) ** (1 / 3)
        b_ = (2 * e1 * ba / math.pi) ** (1 / 3)
        k = a_ / b_
        K = math.pi / ba * E * (r * e1 / (4.5 * e2 ** 3)) ** 0.5
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    Info_hs = (K,
               # Coefficient of load-deformerd.
               k,
               # Value of a/b ratio.
               e1,
               # Elliptic integrals.
               e2,
               # Elliptic integrals.
               E,
               # Equivalent elastic Modulus.
               e,
               # Coefficient of equivalent elastic Modulus.
               rx,
               # Equivalent radius in x direction.
               r,
               # Equivalent radius.
               a_,
               # a star.
               b_
               # b star.
               )

    return Info_hs

###############################################################################
#                        Calculate traction coefficient                       #
###############################################################################
def traction_coefficient(miu_ini, miu_inf, miu_max, um):
    """Solve traction coefficient.

    Parameters
    ----------
    miu_ini: float
        Traction coefficient at zero slip.
    miu_inf: float
        Maximum traction coefficient.
    miu_max: float
        Traction coefficient at infinite slip.
    um: float
        Slip velocity corresponding to maximum traction.

    Returns
    -------
    A: float or np.darray
        Traction coefficient of A.
    B: float or np.darray
        Traction coefficient of B.
    C: float or np.darray
        Traction coefficient of C.
    D: float or np.darray
        Traction coefficient of D.
    """
    A = miu_ini - miu_inf
    D = miu_inf
    solve_B = lambda B: (
        (A + B * um) * math.exp(-B * um / (A + B * um)) + D - miu_max
    )
    B = opt.brentq(solve_B, -1., 1.)
    C = B / (A + B * um)

    return A, B, C, D

###############################################################################
#                  Calculate user-defined traction coefficient                #
###############################################################################
@vectorize(fastmath=False)
def sub_traction_coefficient(ktc, slip, roll, p, a, b, h, tempf, icon):
    """Solve sub traction coefficient.

    Parameters
    ----------
    ktc: float
        lubrication identification.
    slip: float or np.darray
        Sliding velocity.
    roll: float or np.darray
        Rolling velocity.
    p: float or np.darray
        Max value of hertz pressure in contact zone.
    a: float or np.darray
        Major contact half width.
    b: float or np.darray
        Minor contact half width.
    h: float
        Oil thickness.
    tempf: float
        Average surface temperatures.
    icon: float
        Contact identification.

    Returns
    -------
    miu_norm: float or np.darray
        Traction coefficient.
    """
    ###########################################################################
    #              Calculate traction for ball and race contact               #
    ###########################################################################
    if icon == 0:
        # ktc0 = ktc.shape[0]
        cond_0 = np.where(ktc == 1)
        miu_norm = np.zeros_like(ktc)
        #######################################################################
        #              Set boundary lubrication traction number               #
        #######################################################################
        miu_norm[ktc==1] = 0.1
        #######################################################################
        #               Set liquid lubrication traction number                #
        #######################################################################
        """
        A, B, C ,D may be related to p, tempf, and etc.
        need to be developed
        """
        A, B, C, D = 0., 0., 0., 0.
        exp_term = np.exp(-C * np.abs(slip))
        miu_norm[ktc!=1] = (A + B * np.abs(slip)) * exp_term + D
    ###########################################################################
    #              Calculate traction for ball and cage contact               #
    ###########################################################################
    elif icon == 1:
        cond_0 = np.where(ktc == 1)
        miu_norm = np.zeros_like(ktc)
        #######################################################################
        #              Set boundary lubrication traction number               #
        #######################################################################
        miu_norm[ktc==1] = 0.1
        #######################################################################
        #               Set liquid lubrication traction number                #
        #######################################################################
        """
        A, B, C ,D may be related to p, tempf, and etc.
        need to be developed
        """
        A, B, C, D = 0., 0., 0., 0.
        exp_term = np.exp(-C * np.abs(slip))
        miu_norm[ktc!=1] = (A + B * np.abs(slip)) * exp_term + D
    ###########################################################################
    #              Calculate traction for cage and race contact               #
    ###########################################################################
    elif icon == 2:
        cond_0 = np.where(ktc == 1)
        miu_norm = np.zeros_like(ktc)
        #######################################################################
        #              Set boundary lubrication traction number               #
        #######################################################################
        miu_norm[ktc==1] = 0.1
        #######################################################################
        #               Set liquid lubrication traction number                #
        #######################################################################
        """
        A, B, C ,D may be related to p, tempf, and etc.
        need to be developed
        """
        A, B, C, D = 0., 0., 0., 0.
        exp_term = np.exp(-C * np.abs(slip))
        miu_norm[ktc!=1] = (A + B * np.abs(slip)) * exp_term + D
    ###########################################################################
    #              Calculate traction for ball and ball contact               #
    ###########################################################################
    elif icon == 3:
        cond_0 = np.where(ktc == 1)
        miu_norm = np.zeros_like(ktc)
        #######################################################################
        #              Set boundary lubrication traction number               #
        #######################################################################
        miu_norm[ktc==1] = 0.1
        #######################################################################
        #               Set liquid lubrication traction number                #
        #######################################################################
        """
        A, B, C ,D may be related to p, tempf, and etc.
        need to be developed
        """
        A, B, C, D = 0., 0., 0., 0.
        exp_term = np.exp(-C * np.abs(slip))
        miu_norm[ktc!=1] = (A + B * np.abs(slip)) * exp_term + D

    return miu_norm

###############################################################################
#                          Calculate viscous damping                          #
###############################################################################
@vectorize(fastmath=False)
def damping_coefficient(dmpg, ae, m0, m1, k, Q, dtype, ctype):
    """Solve viscous damping.

    Parameters
    ----------
    dmpg: float
        Damping coefficient for damping type II.
    ae: float
        Damping coefficient for damping type I.
    m0: float
        Mass of part I.
    m1: float
        Mass of part II.
    k: np.darray
        Stiffness between the part I and part II.
    Q: np.darray
        Contact force between the part I and part II.
    dtype: int
        Damping type.
    ctype: int
        Cobntact type.

    Returns
    -------
    c: np.darray
        viscous damping between the part I and part II.
    """
    if dtype == 0:
        c = 1.5 * ae * Q
    else:
        m = m0 * m1 / (m0 + m1)
        c = 2 * dmpg * np.sqrt(k * m)

    return c

###############################################################################
#                        Interpolated drag coefficient                        #
###############################################################################
# @njit(fastmath=False)
def drag_coefficient(xi):
    """Interpolated drag coefficient for future using.

    Returns
    -------
    f_drag: callable
        A callable function to predict drag coefficient
        (extrapolation is not allowed).
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    exp_rn = np.array([10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3, 10e4, 2*10e4,
                       3*10e4, 4*10e4, 5*10e4, 10e5])
    exp_dc = np.array([275.00, 30.00, 4.20, 1.20, 0.48, 0.40, 0.45, 0.40, 0.10,
                       0.09, 0.09, 0.09])
    ###########################################################################
    #                   Linear interpolate drag coefficient                   #
    ###########################################################################
    rn = np.log(exp_rn)
    dc = np.log(exp_dc)

    rns0 = rn.shape[0]
    if xi <= rn[0]:
        m0, m1 = 0, 1
    else:
        i = 0
        while xi > rn[i] and i < rns0 - 1:
            i = i + 1
        m0, m1 = i - 1, i
    yi = dc[m0] + (dc[m1] - dc[m0]) * (xi - rn[m0]) / (rn[m1] - rn[m0])

    return yi

###############################################################################
#                      Calculate the oil film thickness                       #
###############################################################################
# @njit(fastmath=False)
def film_thickness(p, a, b, ur, us, rx, ep0, vl, vc0, dl, tcl, sp, ftype):
    """Solve the oil film thickness.

    Parameters
    ----------
    p: float
        Max value of hertz pressure in contact zone.
    a: float
        Major contact half width.
    b: float
        Minor contact half width.
    ur: float
        Roll velocity.
    us: float
        Slip velocity.
    rx: float
        Equivalent radius of deformed surface.
    ep1: float
        Elasticity parameter.
    vl: float
        Base viscosity at current temperature.
    vc0: float
        Pressure-viscosity coefficient at current temperature.
    dl: float
        Denisty at current temperature.
    tcl: float
        Thermal conductivity at current temperature.
    sp: float
        Specific heat at current temperature.
    ftype: float
        Film type.

    Returns
    -------
    hiso: float
        Isothermal film thickness.
    phit: float
        Thermal reduction factor.
    sip: float
        Local variable for starvation eqn.
    """
    ###########################################################################
    #                                Non load                                 #
    ###########################################################################
    if p <= 0:
        hiso = 0.
        if hiso <= 0:
            sip = 0.
            phit = 0.
        return hiso, phit, sip
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    k = a / b
    ep = 2 / ep0
    ee = vc0 * ep
    uu = vl * ur / (ep * rx)
    ww = math.pi * b * p / (2 * ep * rx)
    ###########################################################################
    #                              Point contact                              #
    ###########################################################################
    if ftype > 0:
        hiso = 2 * rx * ee ** 0.6 * uu ** 0.7 / ww ** 0.13
    ###########################################################################
    #                              Line contact                               #
    ###########################################################################
    else:
        hiso = (
            2.69 * rx * uu ** 0.67 * ee ** 0.53 *
            (1 - 0.61 * np.exp(-0.73 * k)) /
            (4 * ww * a / (3 * rx)) ** 0.067
        )
    ###########################################################################
    #                          Starve oil correction                          #
    ###########################################################################
    sip = sp * (b ** 2 / (2 * hiso * rx)) ** (2 / 3)
    if sip < 8:
        hiso = hiso * (1 - np.exp(-1.68005 * sip ** 0.8315 +
                                  0.260137 * sip ** 1.558 -
                                  0.016146 * sip ** 2.296)
        )
    ###########################################################################
    #                             Heat correction                             #
    ###########################################################################
    xll = dl * ur ** 2 / tcl
    phia = 1 - 13.2 * p / ep * xll ** 0.42
    phib = 0.213 * xll ** 0.62
    phit = phia / (1 + phib * (1 + 2.23 * (us / ur) ** 0.83))
    if phit < 0.01:
        phit = 0.01
    return hiso, phit, sip

###############################################################################
#                         Calculate race radius change                        #
###############################################################################
@vectorize(fastmath=False)
def race_radius(x, r0, phi, imp, var0, var1, var2):
    """Solve the race radius change.

    Parameters
    ----------
    x: float
        position in x direction (race fixed frame).
    r0: float
        Base radius.
    phi: np.darray
        Azimuth angle of the race.
    imp: int
        Imperfections of race geometry.
    var0: float
        Deviation of the semi-major axis from nominal or radius.
    var1: float
        Deviation of the semi-minor axis from nominal or radius.
    var2: float
        Orientation (deg) of major axis relative in x direction.

    Returns
    -------
    r: float
        Effective radius.
    """
    ###########################################################################
    #                           Cylindrical radius                            #
    ###########################################################################
    if imp == 0:
        r = r0
    elif imp == 1:
        aa = r0 + var0
        bb = r0 + var1
        r = 1 / np.sqrt(
            (np.cos(phi - var2) / aa) ** 2 +
            (np.sin(phi - var2) / bb) ** 2
        )
    elif imp == 2:
        r = r0 + var0 * np.sin(var1 * phi + var2)

    return r

###############################################################################
#                     Calculate misalignment of the race                      #
###############################################################################
def misalignment(thetay0, thetaz0, thetay1, thetaz1):
    """Solve the misalignment of the race.

    Parameters
    ----------
    thetay0: float
        Misalignment of the outer race in y direction (internal frame).
    thetaz0: float
        Misalignment of the outer race in z direction (internal frame).
    thetay1: float
        Misalignment of the inner race in y direction (internal frame).
    thetaz1: float
        Misalignment of the inner race in z direction (internal frame).

    Returns
    -------
    miso: np.darray
        Transtation from internal to misalignment for outer race.
    misi: np.darray
        Transtation from internal to misalignment for inner race.
    """
    ###########################################################################
    #                   Initial misalignment of outer race                    #
    ###########################################################################
    sin_thty0, cos_thty0 = math.sin(thetay0), math.cos(thetay0)
    sin_thtz0, cos_thtz0 = math.sin(thetaz0), math.cos(thetaz0)

    miso = np.zeros((1, 3, 3))
    miso[0,0,0] = cos_thty0 * cos_thtz0
    miso[0,0,1] = sin_thtz0
    miso[0,0,2] = -sin_thty0 * cos_thtz0
    miso[0,1,0] = -cos_thty0 * sin_thtz0
    miso[0,1,1] = cos_thtz0
    miso[0,1,2] = sin_thty0 * sin_thtz0
    miso[0,2,0] = sin_thty0
    miso[0,2,2] = cos_thty0
    ###########################################################################
    #                   Initial misalignment of inner race                    #
    ###########################################################################
    sin_thty1, cos_thty1 = math.sin(thetay1), math.cos(thetay1)
    sin_thtz1, cos_thtz1 = math.sin(thetaz1), math.cos(thetaz1)

    misi = np.zeros((1, 3, 3))
    misi[0,0,0] = cos_thty1 * cos_thtz1
    misi[0,0,1] = sin_thtz1
    misi[0,0,2] = -sin_thty1 * cos_thtz1
    misi[0,1,0] = -cos_thty1 * sin_thtz1
    misi[0,1,1] = cos_thtz1
    misi[0,1,2] = sin_thty1 * sin_thtz1
    misi[0,2,0] = sin_thty1
    misi[0,2,2] = cos_thty1

    return miso, misi

###############################################################################
#                        Calculate irregular geometry                         #
###############################################################################
def geometry_eccentric(x, y, z, thetax, thetay, thetaz, num, num0, num1):
    """Solve the irregular geometry.

    Parameters
    ----------
    x: float
        Distance form geometery center to mass center in x direction
        (fixed frame).
    y: float
        Distance form geometery center to mass center in y direction
        (fixed frame).
    z: float
        Distance form geometery center to mass center in z direction
        (fixed frame).
    thetax: float
        Angle from geometry matrix to principal in x direction
        (fixed frame).
    thetay: float
        Angle from geometry matrix to principal in y direction
        (fixed frame).
    thetaz: float
        Angle from geometry matrix to principal in z direction
        (fixed frame).
    num: float
        Total number of irregular geometry.
    num0: float
        Interval number of irregular geometry (geometry to mass).
    num1: float
        Interval number of irregular geometry (geometry to principal).

    Returns
    -------
    gg: np.darray
        Geometry center to mass center array (fixed frame).
    gp: np.darray
        Transtation from geometry matrix to principal matrix.
    pg: np.darray
        Transtation from principal matrix to geometry matrix.
    """
    ###########################################################################
    #                     Geometry center to mass center                      #
    ###########################################################################
    gg = np.zeros((num, 3, 1))
    gg[0:num:num0, 0, 0], gg[0:num:num0, 1, 0], gg[0:num:num0, 2, 0] = x, y, z
    ###########################################################################
    #                   Geometry matrix to principal matrix                   #
    ###########################################################################
    sin_thtx, cos_thtx = math.sin(thetax), math.cos(thetax)
    sin_thty, cos_thty = math.sin(thetay), math.cos(thetay)
    sin_thtz, cos_thtz = math.sin(thetaz), math.cos(thetaz)

    gp = np.zeros((num, 3, 3))
    gp[0:num:num1, 0, 0] = cos_thty * cos_thtz
    gp[0:num:num1, 0, 1] = (cos_thtx * sin_thtz +
                            sin_thtx * sin_thty * cos_thtz)
    gp[0:num:num1, 0, 2] = (sin_thtx * sin_thtz -
                            cos_thtx * sin_thty * cos_thtz)
    gp[0:num:num1, 1, 0] = -cos_thty * sin_thtz
    gp[0:num:num1, 1, 1] = (cos_thtx * cos_thtz -
                            sin_thtx * sin_thty * sin_thtz)
    gp[0:num:num1, 1, 2] = (sin_thtx * cos_thtz +
                            cos_thtx * sin_thty * cos_thtz)
    gp[0:num:num1, 2, 0] = sin_thty
    gp[0:num:num1, 2, 1] = -sin_thtx * cos_thty
    gp[0:num:num1, 2, 2] = cos_thtx * cos_thty

    pg = np.transpose(gp, (0,2,1))

    return gg, gp, pg

###############################################################################
#                       Function of slip velocity zero                        #
###############################################################################
# @njit(fastmath=False)
def slip_zero(x, Rp, rb, la, A10, D10, B00, B20, C00, C20, E00, E20, F00, F20):
    """Solve the ball and cage force.

    Parameters
    ----------
    x: float
        argument.
    Rp: float
        Parameter.
    la: float
        Parameter.
    rb: float
        Parameter.
    A10: float
        Parameter.
    D10: float
        Parameter.
    B00: float
        Parameter.
    B20: float
        Parameter.
    C00: float
        Parameter.
    C20: float
        Parameter.
    E00: float
        Parameter.
    E20: float
        Parameter.
    F00: float
        Parameter.
    F20: float
        Parameter.

    Returns
    -------
    f: float
        Dependent variable.
    """
    if Rp ** 2 - (x * la) ** 2 <= 0:
        xx = 0.
    else:
        xx = math.sqrt(Rp ** 2 - (x * la) ** 2)
    z = rb - Rp + xx
    aa, bb, cc, dd = (
        (D10 - A10), (B20 - E20), (E00 - B00), B20 * C00 - B00 * C20
    )

    f = aa + bb * la * x + cc * z + dd + bb * F00 + cc * F20

    return f

###############################################################################
#                     Solve slip velocity zeros equations                     #
###############################################################################
# @njit(fastmath=False)
def solve_slip_zero(x0, x1, Rp, rb, la, A10, D10, B00, B20, C00, C20, E00,
                    E20, F00, F20):
    """Solve the ball and cage force.

    Parameters
    ----------
    x0: float
        One end of the bracketing interval [a,b].
    x1: float
        The other end of the bracketing interval [a,b].
    Rp: float
        Parameter.
    la: float
        Parameter.
    rb: float
        Parameter.
    A10: float
        Parameter.
    D10: float
        Parameter.
    B00: float
        Parameter.
    B20: float
        Parameter.
    C00: float
        Parameter.
    C20: float
        Parameter.
    E00: float
        Parameter.
    E20: float
        Parameter.
    F00: float
        Parameter.
    F20: float
        Parameter.

    Returns
    -------
    b: float
        Zero of f between a and b.
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    a, b, c = x0, x1, x1
    fa, fb, fc = (
        slip_zero(
            a, Rp, rb, la, A10, D10, B00, B20, C00, C20, E00, E20, F00, F20
        ),
        slip_zero(
            b, Rp, rb, la, A10, D10, B00, B20, C00, C20, E00, E20, F00, F20
        ),
        slip_zero(
            c, Rp, rb, la, A10, D10, B00, B20, C00, C20, E00, E20, F00, F20
        )
    )
    if (fa > 0 and fb > 0) or (fa < 0 and fb < 0):
        return math.nan
    fc = fb
    ###########################################################################
    #                                  Loop                                   #
    ###########################################################################
    for i in range(100):
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c, fc = a, fa; e = d = b - a
        if abs(fc) < abs(fb):
            a = b; b = c; c = a
            fa = fb; fb = fc; fc = fa
        tol0 = 2 * 1e-12 * abs(b) + 0.5 * 8e-16
        xm = 0.5 * (c - b)
        if (abs(xm) <= tol0) or (fb == 0):
            return b
        if (abs(e) >= tol0) and (abs(fa) > abs(fb)):
            s = fb / fa
            if a == c:
                p = 2 * xm * s; q = 1 - s
            else:
                q = fa / fc; r = fb / fc
                p = s * (2 * xm * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            if p > 0:
                q = -q
            p = abs(p)
            min0 = 3 * xm * q - abs(tol0 * q)
            min1 = abs(e * q)
            if min0 < min1:
                min2 = min0
            else:
                min2 = min1
            if (2 * p < min2):
                e = d; d = p / q
            else:
                d = xm; e = d
        else:
            d = xm; e = d
        a = b; fa = fb
        if abs(d) > tol0:
            b += d
        else:
            if xm < 0:
                tol1 = -tol0
            else:
                tol1 = tol0
            b += tol1
        fb = slip_zero(
            b, Rp, rb, la, A10, D10, B00, B20, C00, C20, E00, E20, F00, F20
        )
    return b

###############################################################################
#       Calculate the transformation relationship between ball and race       #
###############################################################################
# @njit(fastmath=False)
def ball_race_contact_strain(x, Info_es, mod_brcs):
    """Solve the transformation relationship between ball and race.

    Parameters
    ----------
    x: np.darray
        Solution vector.
    Info_es: tuple
        Information of expansion_subcall.
    mod_brcs: tuple
        Mode data of ball_race_contact_strain.

    Returns
    -------
    delta_b_o: np.darray
        Strain between the ball and outer race.
    delta_b_i: np.darray
        Strain between the ball and inner race.
    Info_brcs: tuple
        Information of ball_race_contact_strain.
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    (D_b,
     D_m,
     T_I_imis,
     T_I_omis,
     T_bp_b,
     Shim_thknss_i,
     Shim_thknss_o,
     f_i,
     f_o,
     free_con_ang,
     k_geo_imc_type_i,
     k_geo_imc_type_o,
     r_bg_bm_b,
     n,
     var_i_r0,
     var_i_r1,
     var_i_r2,
     var_o_r0,
     var_o_r1,
     var_o_r2
     ) = mod_brcs[0::]
    ###########################################################################
    #                               End prepare                               #
    ###########################################################################
    race_exp_o, race_exp_i = Info_es[9:11]

    cos_free_con_ang = math.cos(free_con_ang)
    P_rad_o_term = 0.5 * (D_m - 2 * (f_o - 0.5) * D_b * cos_free_con_ang)
    P_rad_o = P_rad_o_term + race_exp_o

    P_rad_i_term = 0.5 * (D_m + 2 * (f_i - 0.5) * D_b * cos_free_con_ang)
    P_rad_i = P_rad_i_term + race_exp_i

    sin_vals = np.sin(x[28:24+12*n:12])
    cos_vals = np.cos(x[28:24+12*n:12])

    T_I_a = np.zeros((n, 3, 3))
    T_I_a[:, 0, 0] = 1.
    T_I_a[:, 1, 1] = cos_vals
    T_I_a[:, 1, 2] = sin_vals
    T_I_a[:, 2, 1] = -T_I_a[:, 1, 2]
    T_I_a[:, 2, 2] = T_I_a[:, 1, 1]

    T_a_I = np.transpose(T_I_a, (0,2,1))

    cos_x30 = np.cos(x[30:24+12*n:12])
    sin_x30 = np.sin(x[30:24+12*n:12])
    cos_x32 = np.cos(x[32:24+12*n:12])
    sin_x32 = np.sin(x[32:24+12*n:12])
    cos_x34 = np.cos(x[34:24+12*n:12])
    sin_x34 = np.sin(x[34:24+12*n:12])

    T_a_bp = np.zeros((n, 3, 3))
    T_a_bp[:, 0, 0] = cos_x32 * cos_x34
    T_a_bp[:, 0, 1] = cos_x30 * sin_x34 + sin_x30 * sin_x32 * cos_x34
    T_a_bp[:, 0, 2] = sin_x30 * sin_x34 - cos_x30 * sin_x32 * cos_x34
    T_a_bp[:, 1, 0] = -cos_x32 * sin_x34
    T_a_bp[:, 1, 1] = cos_x30 * cos_x34 - sin_x30 * sin_x32 * sin_x34
    T_a_bp[:, 1, 2] = sin_x30 * cos_x34 + cos_x30 * sin_x32 * sin_x34
    T_a_bp[:, 2, 0] = sin_x32
    T_a_bp[:, 2, 1] = -sin_x30 * cos_x32
    T_a_bp[:, 2, 2] = cos_x30 * cos_x32

    T_a_b = np.zeros((n, 3, 3))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 3))

    for i in range(n):
        t0[:, :] = T_bp_b[i, :, :]
        t1[:, :] = T_a_bp[i, :, :]
        t2 = np.dot(t0, t1)
        T_a_b[i, :, :] = t2

    T_I_b = np.zeros((n,3,3))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 3))

    for i in range(n):
        t0[:, :] = T_a_b[i, :, :]
        t1[:, :] = T_I_a[i, :, :]
        t2 = np.dot(t0, t1)
        T_I_b[i, :, :] = t2

    T_b_I = np.transpose(T_I_b, (0,2,1))

    cos_x30_n = np.cos(x[30+12*n])
    sin_x30_n = np.sin(x[30+12*n])
    cos_x32_n = np.cos(x[32+12*n])
    sin_x32_n = np.sin(x[32+12*n])
    cos_x34_n = np.cos(x[34+12*n])
    sin_x34_n = np.sin(x[34+12*n])

    T_omis_o = np.array([[
        [cos_x32_n * cos_x34_n,
         cos_x30_n * sin_x34_n + sin_x30_n * sin_x32_n * cos_x34_n,
         sin_x30_n * sin_x34_n - cos_x30_n * sin_x32_n * cos_x34_n],
        [-cos_x32_n * sin_x34_n,
         cos_x30_n * cos_x34_n - sin_x30_n * sin_x32_n * sin_x34_n,
         sin_x30_n * cos_x34_n + cos_x30_n * sin_x32_n * sin_x34_n],
        [sin_x32_n,
         -sin_x30_n * cos_x32_n,
         cos_x30_n * cos_x32_n]
    ]])

    T_I_o = np.zeros((1, 3, 3))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 3))

    for i in range(1):
        t0[:, :] = T_omis_o[i, :, :]
        t1[:, :] = T_I_omis[i, :, :]
        t2 = np.dot(t0, t1)
        T_I_o[i, :, :] = t2

    T_o_I = np.transpose(T_I_o, (0,2,1))

    # T_b_o = T_I_o @ T_b_I

    cos_x6 = np.cos(x[6])
    sin_x6 = np.sin(x[6])
    cos_x8 = np.cos(x[8])
    sin_x8 = np.sin(x[8])
    cos_x10 = np.cos(x[10])
    sin_x10 = np.sin(x[10])

    T_imis_i = np.array([[
        [cos_x8 * cos_x10,
         cos_x6 * sin_x10 + sin_x6 * sin_x8 * cos_x10,
         sin_x6 * sin_x10 - cos_x6 * sin_x8 * cos_x10],
        [-cos_x8 * sin_x10,
         cos_x6 * cos_x10 - sin_x6 * sin_x8 * sin_x10,
         sin_x6 * cos_x10 + cos_x6 * sin_x8 * sin_x10],
        [sin_x8, -sin_x6 * cos_x8, cos_x6 * cos_x8]
    ]])

    T_I_i = np.zeros((1, 3, 3))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 3))

    for i in range(1):
        t0[:, :] = T_imis_i[i, :, :]
        t1[:, :] = T_I_imis[i, :, :]
        t2 = np.dot(t0, t1)
        T_I_i[i, :, :] = t2

    T_i_I = np.transpose(T_I_i, (0,2,1))

    # T_b_i = T_I_i @ T_b_I

    r_bm_I = np.zeros((n, 3, 1))
    r_bm_I[:, 0, 0] = x[24:24+12*n:12]
    r_bm_I[:, 1, 0] = -x[26:24+12*n:12] * sin_vals
    r_bm_I[:, 2, 0] = x[26:24+12*n:12] * cos_vals

    r_bg_I = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 1))
    t2 = np.zeros((3, 1))
    t3 = np.zeros((3, 1))
    t4 = np.zeros((3, 1))

    for i in range(n):
        t0[:, :] = T_b_I[i, :, :]
        t1[:, :] = r_bg_bm_b[i, :, :]
        t2 = np.dot(t0, t1)
        t3[:, :] = r_bm_I[i, :, :]
        t4 = t3 + t2
        r_bg_I[i, :, :] = t4
    # ALmost ignore of the orientation of rolling element principal axes

    r_om_I = np.zeros((1, 3, 1))
    r_om_I[0, 0, 0] = x[24+12*n]
    r_om_I[0, 1, 0] = x[26+12*n]
    r_om_I[0, 2, 0] = x[28+12*n]
    """
    Outer race geo center need to be developed
    """
    r_og_I = np.copy(r_om_I)

    r_im_I = np.zeros((1, 3, 1))
    r_im_I[0, 0, 0] = x[0]
    r_im_I[0, 1, 0] = x[2]
    r_im_I[0, 2, 0] = x[4]
    """
    Inner race geo center need to be developed
    """
    r_ig_I = np.copy(r_im_I)
    ###########################################################################
    #         Transformation relationship between ball and outer race         #
    ###########################################################################
    r_bg_og_o = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 1))

    for i in range(n):
        t0[:, :] = T_I_o[0, :, :]
        t1[:, :] = (r_bg_I - r_og_I)[i, :, :]
        t2 = np.dot(t0, t1)
        r_bg_og_o[i, :, :] = t2

    phi_b_o = np.zeros((n, 1, 1))
    phi_b_o[:, 0, 0] = np.arctan2(-r_bg_og_o[:, 1, 0], r_bg_og_o[:, 2, 0])

    e_o = race_radius(
        0, P_rad_o, phi_b_o, k_geo_imc_type_o, var_o_r0, var_o_r1, var_o_r2
        )

    r_ogc_o = np.zeros((n, 3, 1))
    r_ogc_o[:, 0, 0] = -Shim_thknss_o * 0.5
    r_ogc_o[:, 1, 0] = -e_o[:, 0, 0] * np.sin(phi_b_o[:, 0, 0])
    r_ogc_o[:, 2, 0] = e_o[:, 0, 0] * np.cos(phi_b_o[:, 0, 0])

    r_bg_ogc_a = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 1))
    t2 = np.zeros((3, 1))
    t3 = np.zeros((3, 3))
    t4 = np.zeros((3, 1))
    t5 = np.zeros((3, 1))

    for i in range(n):
        t0[:, :] = T_o_I[0, :, :]
        t1[:, :] = r_ogc_o[i, :, :]
        t2 = np.dot(t0, t1)
        t3[:, :] = T_I_a[i, :, :]
        t4[:, :] = (r_bg_I[i, :, :] - r_og_I[0, :, :] - t2)
        t5 = np.dot(t3, t4)
        r_bg_ogc_a[i, :, :] = t5

    r_bg_ogc_a_norm = np.zeros((n, 1, 1))
    r_bg_ogc_a_norm[:, 0, 0] = np.sqrt(r_bg_ogc_a[:, 0, 0] ** 2 +
                                       r_bg_ogc_a[:, 1, 0] ** 2 +
                                       r_bg_ogc_a[:, 2, 0] ** 2)

    e_bg_ogc_a = r_bg_ogc_a / r_bg_ogc_a_norm

    delta_b_o = r_bg_ogc_a_norm - (f_o - 0.5) * D_b
    for i in range(n):
        if delta_b_o[i, 0, 0] < 1e-10:
            delta_b_o[i, 0, 0] = 0.

    alpha_b_o_0 = np.zeros((n, 1, 1))
    alpha_b_o_0[:, 0, 0] = np.arctan2(e_bg_ogc_a[:, 0, 0], e_bg_ogc_a[:, 2, 0])
    for i in range(n):
        if np.abs(e_bg_ogc_a[i, 0, 0] / e_bg_ogc_a[i, 2, 0]) < 1e-6:
            alpha_b_o_0[i, 0, 0] = 0.

    alpha_b_o_1 = np.zeros((n, 1, 1))
    alpha_b_o_1[:, 0, 0] = np.arctan2(
        -e_bg_ogc_a[:, 1, 0],
        np.hypot(e_bg_ogc_a[:, 0, 0], e_bg_ogc_a[:, 2, 0])
    )
    if np.abs(e_bg_ogc_a[i, 1, 0]) < 1e-5:
        alpha_b_o_1[i, 0, 0] = 0.

    cos_alpha_0 = np.cos(alpha_b_o_0[:, 0, 0])
    sin_alpha_0 = np.sin(alpha_b_o_0[:, 0, 0])
    cos_alpha_1 = np.cos(alpha_b_o_1[:, 0, 0])
    sin_alpha_1 = np.sin(alpha_b_o_1[:, 0, 0])

    T_a_oc = np.zeros((n, 3, 3))
    T_a_oc[:, 0, 0] = cos_alpha_0
    T_a_oc[:, 0, 2] = -sin_alpha_0
    T_a_oc[:, 1, 0] = sin_alpha_0 * sin_alpha_1
    T_a_oc[:, 1, 1] = cos_alpha_1
    T_a_oc[:, 1, 2] = cos_alpha_0 * sin_alpha_1
    T_a_oc[:, 2, 0] = sin_alpha_0 * cos_alpha_1
    T_a_oc[:, 2, 1] = -sin_alpha_1
    T_a_oc[:, 2, 2] = cos_alpha_0 * cos_alpha_1
    ###########################################################################
    #         Transformation relationship between ball and inner race         #
    ###########################################################################
    r_bg_ig_i = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 1))

    for i in range(n):
        t0[:, :] = T_I_i[0,:,:]
        t1[:, :] = (r_bg_I - r_ig_I)[i, :, :]
        t2 = np.dot(t0, t1)
        r_bg_ig_i[i, :, :] = t2

    phi_b_i = np.zeros((n, 1, 1))
    phi_b_i[:, 0, 0] = np.arctan2(-r_bg_ig_i[:, 1, 0], r_bg_ig_i[:, 2, 0])

    e_i = race_radius(
        0, P_rad_i, phi_b_i, k_geo_imc_type_i, var_i_r0, var_i_r1, var_i_r2
        )

    r_igc_i = np.zeros((n, 3, 1))
    r_igc_i[:, 0, 0] = Shim_thknss_i * 0.5
    r_igc_i[:, 1, 0] = -e_i[:, 0, 0] * np.sin(phi_b_i[:, 0, 0])
    r_igc_i[:, 2, 0] = e_i[:, 0, 0] * np.cos(phi_b_i[:, 0, 0])

    r_bg_igc_a = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 1))
    t2 = np.zeros((3, 1))
    t3 = np.zeros((3, 3))
    t4 = np.zeros((3, 1))
    t5 = np.zeros((3, 1))

    for i in range(0, n):
        t0[:, :] = T_i_I[0, :, :]
        t1[:, :] = r_igc_i[i, :, :]
        t2 = np.dot(t0, t1)
        t3[:, :] = T_I_a[i, :, :]
        t4[:, :] = (r_bg_I[i, :, :] - r_ig_I[0, :, :] - t2)
        t5 = np.dot(t3, t4)
        r_bg_igc_a[i, :, :] = t5

    r_bg_igc_a_norm = np.zeros((n, 1, 1))
    r_bg_igc_a_norm[:, 0, 0] = np.sqrt(r_bg_igc_a[:, 0, 0] ** 2 +
                                       r_bg_igc_a[:, 1, 0] ** 2 +
                                       r_bg_igc_a[:, 2, 0] ** 2)


    e_bg_igc_a = r_bg_igc_a / r_bg_igc_a_norm

    delta_b_i = r_bg_igc_a_norm - (f_i - 0.5) * D_b
    for i in range(n):
        if delta_b_i[i, 0, 0] < 1e-10:
            delta_b_i[i, 0, 0] = 0.

    alpha_b_i_0 = np.zeros((n, 1, 1))
    alpha_b_i_0[:, 0, 0] = np.arctan2(e_bg_igc_a[:, 0, 0], e_bg_igc_a[:, 2, 0])
    for i in range(n):
        if np.abs(e_bg_igc_a[i, 0, 0] / e_bg_igc_a[i, 2, 0]) < 1e-6:
            alpha_b_i_0[i, 0, 0] = 0.

    alpha_b_i_1 = np.zeros((n, 1, 1))
    alpha_b_i_1[:, 0, 0] = np.arctan2(
        -e_bg_igc_a[:, 1, 0],
        np.hypot(e_bg_igc_a[:, 0, 0], e_bg_igc_a[:, 2, 0])
    )
    if np.abs(e_bg_igc_a[i, 1, 0]) < 1e-5:
        alpha_b_i_1[i, 0, 0] = 0.

    cos_alpha_0 = np.cos(alpha_b_i_0[:, 0, 0])
    sin_alpha_0 = np.sin(alpha_b_i_0[:, 0, 0])
    cos_alpha_1 = np.cos(alpha_b_i_1[:, 0, 0])
    sin_alpha_1 = np.sin(alpha_b_i_1[:, 0, 0])

    T_a_ic = np.zeros((n, 3, 3))
    T_a_ic[:, 0, 0] = cos_alpha_0
    T_a_ic[:, 0, 2] = -sin_alpha_0
    T_a_ic[:, 1, 0] = sin_alpha_0 * sin_alpha_1
    T_a_ic[:, 1, 1] = cos_alpha_1
    T_a_ic[:, 1, 2] = cos_alpha_0 * sin_alpha_1
    T_a_ic[:, 2, 0] = sin_alpha_0 * cos_alpha_1
    T_a_ic[:, 2, 1] = -sin_alpha_1
    T_a_ic[:, 2, 2] = cos_alpha_0 * cos_alpha_1
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    Info_brcs = (T_o_I,
                 # Outer race to inertial transformation matrix.
                 T_I_o,
                 # Inertial to outer race transformation matrix.
                 T_i_I,
                 # Inner race to inertial transformation matrix.
                 T_I_i,
                 # Inertial to inner race transformation matrix.
                 T_a_I,
                 # Azimuth to inertial transformation matrix.
                 T_I_a,
                 # Inertial to azimuth transformation matrix.
                 T_b_I,
                 # Ball to inertial transformation matrix.
                 T_I_b,
                 # Inertial to ball transformation matrix.
                 T_a_oc,
                 # Inertial azimuth to outer race contact frame transformation
                 # matrix.
                 T_a_ic,
                 # Inertial azimuth to inner race contact frame transformation
                 # matrix.
                 r_om_I,
                 # Outer race mass center in intertial frame.
                 r_og_I,
                 # Outer race geometry center in intertial frame.
                 r_im_I,
                 # Inner race mass center in intertial frame.
                 r_ig_I,
                 # Inner race geometry center in intertial frame.
                 r_bm_I,
                 # Ball mass center in intertial frame.
                 r_bg_I,
                 # Ball geometry center in intertial frame.
                 e_o,
                 # Race curve radius of outer race.
                 e_i,
                 # Race curve radius of inner race.
                 phi_b_o,
                 # Race azimuth angle of outer race.
                 phi_b_i,
                 # Race azimuth angle of inner race.
                 alpha_b_o_0,
                 # Contact angle between ball and outer race.
                 alpha_b_o_1,
                 # Yaw angle between ball and outer race.
                 alpha_b_i_0,
                 # Contact angle between ball and inner race.
                 alpha_b_i_1
                 # Yaw angle between ball and inner race.
                 )

    return delta_b_o, delta_b_i, Info_brcs

###############################################################################
#  Calculate the transformation relationship between ball and secondary race  #
###############################################################################
# @njit(fastmath=False)
def ball_race_contact_strain_(Info_brcs, mod_brcs_):
    """Solve the transformation relationship between ball and race.

    Parameters
    ----------
    x: np.darray
        Solution vector.
    Info_brcs: tuple
        Information of ball_race_contact_strain.
    mod_brcs_: tuple
        Mode data of ball_race_contact_strain_.

    Returns
    -------
    delta_b_o_: np.darray
        Strain between the ball and secondary outer race.
    delta_b_i_: np.darray
        Strain between the ball and secondary inner race.
    Info_brcs_: tuple
        Information of ball_race_contact_strain_.
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    D_b, Shim_thknss_i, Shim_thknss_o, f_i, f_o, max_rs, n = mod_brcs_[0::]
    ###########################################################################
    #                            No secondary race                            #
    ###########################################################################
    if max_rs == 2:
        (
            delta_b_o_, delta_b_i_, T_a_oc_, T_a_ic_, nor_dis_b_o_,
            nor_dis_b_i_, alpha_b_o_0_, alpha_b_i_0_, alpha_b_o_1_,
            alpha_b_i_1_, num_con_b_o_, num_con_b_i_
            ) = (
                np.zeros((n,1,1)), np.zeros((n,1,1)), np.zeros((n,3,3)),
                np.zeros((n,3,3)), np.zeros((n,1,1)), np.zeros((n,1,1)),
                np.zeros((n,1,1)), np.zeros((n,1,1)), np.zeros((n,1,1)),
                np.zeros((n,1,1)), 0, 0
                )
    
        Info_brcs_ = (T_a_oc_,
                      # Azimuth to secondary outer race contact frame
                      # transformation matrix.
                      T_a_ic_,
                      # Azimuth to secondary inner race contact frame
                      # transformation matrix.
                      nor_dis_b_o_,
                      # Normal distance between ball and secondary outer race.
                      nor_dis_b_i_,
                      # Normal distance between ball and secondary inner race.
                      alpha_b_o_0_,
                      # Contact angle between ball and secondary outer race.
                      alpha_b_o_1_,
                      # Yaw angle between ball and secondary outer race.
                      alpha_b_i_0_,
                      # Contact angle between ball and secondary inner race.
                      alpha_b_i_1_,
                      # Yaw angle between ball and secondary inner race.
                      num_con_b_o_,
                      # Contact number between ball and secondary outer race.
                      num_con_b_i_,
                      # Contact number between ball and secondary inner race.
                      )
        return delta_b_o_, delta_b_i_, Info_brcs_
    ###########################################################################
    #               Secondary outer race for shim thickness > 0               #
    ###########################################################################
    if Shim_thknss_o > 0:
        #######################################################################
        #                           For all balls                             #
        #######################################################################
        (
            T_o_I, T_I_a, r_og_I, r_bg_I, e_o_, phi_b_o_
            ) = (
                Info_brcs[0], Info_brcs[5], Info_brcs[11],
                Info_brcs[15], Info_brcs[16], Info_brcs[18]
                )

        r_ogc_o_ = np.zeros((n, 3, 1))
        r_ogc_o_[:, 0, 0] = Shim_thknss_o * 0.5
        r_ogc_o_[:, 1, 0] = -e_o_[:, 0, 0] * np.sin(phi_b_o_[:, 0, 0])
        r_ogc_o_[:, 2, 0] = e_o_[:, 0, 0] * np.cos(phi_b_o_[:, 0, 0])

        r_bg_ogc_a_ = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))
        t2 = np.zeros((3, 1))
        t3 = np.zeros((3, 3))
        t4 = np.zeros((3, 1))
        t5 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_o_I[0, :, :]
            t1[:, :] = r_ogc_o_[i, :, :]
            t2 = np.dot(t0, t1)
            t3[:, :] = T_I_a[i, :, :]
            t4[:, :] = (r_bg_I[i, :, :] - r_og_I[0, :, :] - t2)
            t5 = np.dot(t3, t4)
            r_bg_ogc_a_[i, :, :] = t5

        r_bg_ogc_a_norm_ = np.zeros((n, 1, 1))
        r_bg_ogc_a_norm_[:, 0, 0] = np.sqrt(r_bg_ogc_a_[:, 0, 0] ** 2 +
                                            r_bg_ogc_a_[:, 1, 0] ** 2 +
                                            r_bg_ogc_a_[:, 2, 0] ** 2)

        e_bg_ogc_a_ = r_bg_ogc_a_ / r_bg_ogc_a_norm_

        alpha_b_o_0_ = np.zeros((n, 1, 1))
        alpha_b_o_0_[:, 0, 0] = np.arctan2(
            e_bg_ogc_a_[:, 0, 0], e_bg_ogc_a_[:, 2, 0]
        )
        for i in range(n):
            if np.abs(e_bg_ogc_a_[i, 0, 0]/e_bg_ogc_a_[i, 2, 0]) < 1e-6:
                alpha_b_o_0_[i, 0, 0] = 0.

        alpha_b_o_1_ = np.zeros((n, 1, 1))
        alpha_b_o_1_[:, 0, 0] = np.arctan2(
            -e_bg_ogc_a_[:, 1, 0], np.hypot(
                e_bg_ogc_a_[:, 0, 0], e_bg_ogc_a_[:, 2, 0])
        )
        for i in range(n):
            if np.abs(e_bg_ogc_a_[i, 1, 0]) < 1e-5:
                alpha_b_o_1_[i, 0, 0] = 0

        delta_b_o_ = r_bg_ogc_a_norm_ - (f_o - 0.5) * D_b
        nor_dis_b_o_ = np.copy(delta_b_o_)
        for i in range(n):
            if delta_b_o_[i, 0, 0] < 1e-10:
                delta_b_o_[i, 0, 0] = 0

        cond_0 = np.where(delta_b_o_ > 0)
        num_con_b_o_ = len(cond_0[0])
        #######################################################################
        #                     Only for contact zone balls                     #
        #######################################################################
        cos_alpha_0_ = np.cos(alpha_b_o_0_[:, 0, 0])
        sin_alpha_0_ = np.sin(alpha_b_o_0_[:, 0, 0])
        cos_alpha_1_ = np.cos(alpha_b_o_1_[:, 0, 0])
        sin_alpha_1_ = np.sin(alpha_b_o_1_[:, 0, 0])

        T_a_oc_ = np.zeros((n, 3, 3))
        T_a_oc_[:, 0, 0] = cos_alpha_0_
        T_a_oc_[:, 0, 2] = -sin_alpha_0_
        T_a_oc_[:, 1, 0] = sin_alpha_0_ * sin_alpha_1_
        T_a_oc_[:, 1, 1] = cos_alpha_1_
        T_a_oc_[:, 1, 2] = cos_alpha_0_ * sin_alpha_1_
        T_a_oc_[:, 2, 0] = sin_alpha_0_ * cos_alpha_1_
        T_a_oc_[:, 2, 1] = -sin_alpha_1_
        T_a_oc_[:, 2, 2] = cos_alpha_0_ * cos_alpha_1_
    else:
        (
            delta_b_o_, T_a_oc_,
            nor_dis_b_o_, alpha_b_o_0_,
            alpha_b_o_1_, num_con_b_o_
            ) = (
                np.zeros((n, 1, 1)), np.zeros((n, 3, 3)),
                np.zeros((n, 1, 1)), np.zeros((n, 1, 1)),
                np.zeros((n, 1, 1)), 0
                )
    ###########################################################################
    #               Secondary inner race for shim thickness > 0               #
    ###########################################################################
    if Shim_thknss_i > 0:
        #######################################################################
        #                           For all balls                             #
        #######################################################################
        (
            T_i_I, T_I_a, r_ig_I, r_bg_I, e_i_, phi_b_i_
            ) = (
                Info_brcs[2], Info_brcs[5], Info_brcs[13],
                Info_brcs[15], Info_brcs[17], Info_brcs[19]
                )

        r_igc_i_ = np.zeros((n, 3, 1))
        r_igc_i_[:, 0, 0] = -Shim_thknss_i * 0.5
        r_igc_i_[:, 1, 0] = -e_i_[:, 0, 0] * np.sin(phi_b_i_[:, 0, 0])
        r_igc_i_[:, 2, 0] = e_i_[:, 0, 0] * np.cos(phi_b_i_[:, 0, 0])

        r_bg_igc_a_ = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))
        t2 = np.zeros((3, 1))
        t3 = np.zeros((3, 3))
        t4 = np.zeros((3, 1))
        t5 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_i_I[0, :, :]
            t1[:, :] = r_igc_i_[i, :, :]
            t2 = np.dot(t0, t1)
            t3[:, :] = T_I_a[i, :, :]
            t4[:, :] = (r_bg_I[i, :, :] - r_ig_I[0, :, :] - t2)
            t5 = np.dot(t3, t4)
            r_bg_igc_a_[i, :, :] = t5

        r_bg_igc_a_norm_ = np.zeros((n, 1, 1))
        r_bg_igc_a_norm_[:, 0, 0] = np.sqrt(r_bg_igc_a_[:, 0, 0] ** 2 +
                                            r_bg_igc_a_[:, 1, 0] ** 2 +
                                            r_bg_igc_a_[:, 2, 0] ** 2)

        e_bg_igc_a_ = r_bg_igc_a_ / r_bg_igc_a_norm_

        alpha_b_i_0_ = np.zeros((n, 1, 1))
        alpha_b_i_0_[:, 0, 0] = np.arctan2(
            e_bg_igc_a_[:, 0, 0], e_bg_igc_a_[:, 2, 0]
        )
        for i in range(n):
            if np.abs(e_bg_igc_a_[i, 0, 0] / e_bg_igc_a_[i, 2, 0]) < 1e-6:
                alpha_b_i_0_[i, 0, 0] = 0.

        alpha_b_i_1_ = np.zeros((n, 1, 1))
        alpha_b_i_1_[:, 0, 0] = np.arctan2(
            -e_bg_igc_a_[:, 1, 0], np.hypot(
                e_bg_igc_a_[:, 0, 0], e_bg_igc_a_[:, 2, 0])
        )
        for i in range(n):
            if np.abs(e_bg_igc_a_[i, 1, 0]) < 1e-5:
                alpha_b_i_1_[i, 0, 0] = 0.

        delta_b_i_ = r_bg_igc_a_norm_ - (f_i - 0.5) * D_b
        nor_dis_b_i_ = np.copy(delta_b_i_)
        for i in range(n):
            if delta_b_i_[i, 0, 0] < 1e-10:
                delta_b_i_[i, 0, 0] = 0.

        cond_1 = np.where(delta_b_i_ > 0)
        num_con_b_i_ = len(cond_1[0])
        #######################################################################
        #                     Only for contact zone balls                     #
        #######################################################################
        cos_alpha_0_ = np.cos(alpha_b_i_0_[:, 0, 0])
        sin_alpha_0_ = np.sin(alpha_b_i_0_[:, 0, 0])
        cos_alpha_1_ = np.cos(alpha_b_i_1_[:, 0, 0])
        sin_alpha_1_ = np.sin(alpha_b_i_1_[:, 0, 0])

        T_a_ic_ = np.zeros((n, 3, 3))
        T_a_ic_[:, 0, 0] = cos_alpha_0_
        T_a_ic_[:, 0, 2] = -sin_alpha_0_
        T_a_ic_[:, 1, 0] = sin_alpha_0_ * sin_alpha_1_
        T_a_ic_[:, 1, 1] = cos_alpha_1_
        T_a_ic_[:, 1, 2] = cos_alpha_0_ * sin_alpha_1_
        T_a_ic_[:, 2, 0] = sin_alpha_0_ * cos_alpha_1_
        T_a_ic_[:, 2, 1] = -sin_alpha_1_
        T_a_ic_[:, 2, 2] = cos_alpha_0_ * cos_alpha_1_
    else:
        (
            delta_b_i_, T_a_ic_,
            nor_dis_b_i_, alpha_b_i_0_,
            alpha_b_i_1_, num_con_b_i_
            ) = (
                np.zeros((n, 1, 1)), np.zeros((n, 3, 3)),
                np.zeros((n, 1, 1)), np.zeros((n, 1, 1)),
                np.zeros((n, 1, 1)), 0
                )
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    Info_brcs_ = (T_a_oc_,
                  # Azimuth to secondary outer race contact frame
                  # transformation matrix.
                  T_a_ic_,
                  # Azimuth to secondary inner race contact frame
                  # transformation matrix.
                  nor_dis_b_o_,
                  # Normal distance between ball and secondary outer race.
                  nor_dis_b_i_,
                  # Normal distance between ball and secondary inner race.
                  alpha_b_o_0_,
                  # Contact angle between ball and secondary outer race.
                  alpha_b_o_1_,
                  # Yaw angle between ball and secondary outer race.
                  alpha_b_i_0_,
                  # Contact angle between ball and secondary inner race.
                  alpha_b_i_1_,
                  # Yaw angle between ball and secondary inner race.
                  num_con_b_o_,
                  # Contact number between ball and secondary outer race.
                  num_con_b_i_,
                  # Contact number between ball and secondary inner race.
                  )
    return delta_b_o_, delta_b_i_, Info_brcs_

###############################################################################
#           Calculate the normal contact load between ball and race           #
###############################################################################
# @njit(fastmath=False)
def ball_race_contact_force(delta_b_o, delta_b_i, mod_brcf):
    """Solve the transformation relationship between ball and race.

    Parameters
    ----------
    delta_b_o: np.darray
        Strain between the ball and outer race.
    delta_b_i: np.darray
        Strain between the ball and inner race.
    mod_brcf: tuple
        Mode data of ball_race_contact_force.

    Returns
    -------
    Q_b_o: np.darray
        Contact force between the ball and outer race of
        ball_race_contact_force.
    Q_b_i: np.darray
        Contact force between the ball and inner race of
        ball_race_contact_force.
    Info_brcf: tuple
        Information of ball_race_contact_force.
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    (E_b_i,
     E_b_o,
     K_b_i,
     K_b_o,
     R_b,
     R_b_i,
     R_b_o,
     de_b_i,
     de_b_o,
     ke_b_i,
     ke_b_o,
     n
     ) = mod_brcf[0::]
    ###########################################################################
    #         Calculate the normal contact load between ball and race         #
    ###########################################################################
    Q_b_o = K_b_o * delta_b_o ** 1.5

    q_b_o = np.zeros((n, 1, 1))
    for i in range(n):
        if Q_b_o[i, 0, 0] > 0:
            q_b_o[i, 0, 0] = Q_b_o[i, 0, 0] / delta_b_o[i, 0, 0]
    ###########################################################################
    #         Calculate the normal contact shape between ball and race        #
    ###########################################################################
    coeff_aa_b_o = (
            (6 * ke_b_o ** 2 * de_b_o * R_b_o / (math.pi * E_b_o)) ** (1 / 3)
    )
    aa_b_o = (Q_b_o ** (1 / 3)) * coeff_aa_b_o
    for i in range(n):
        if aa_b_o[i, 0, 0] > 0.9 * R_b:
            aa_b_o[i, 0, 0] = 0.9 * R_b

    bb_b_o = aa_b_o / ke_b_o
    ###########################################################################
    #      Calculate the maximum compressive stress in the contact area       #
    ###########################################################################
    p_b_o_max =np.zeros((n, 1, 1))
    for i in range(n):
        if Q_b_o[i, 0, 0] > 0:
            p_b_o_max[i, 0, 0] = (
                1.5 * Q_b_o[i, 0, 0] /
                (math.pi * aa_b_o[i, 0, 0] * bb_b_o[i, 0, 0])
            )
    ###########################################################################
    #         Calculate the normal contact load between ball and race         #
    ###########################################################################
    Q_b_i = K_b_i * delta_b_i ** 1.5

    q_b_i = np.zeros((n, 1, 1))
    for i in range(n):
        if Q_b_i[i, 0, 0] > 0:
            q_b_i[i, 0, 0] = Q_b_i[i, 0, 0] / delta_b_i[i, 0, 0]
    ###########################################################################
    #         Calculate the normal contact shape between ball and race        #
    ###########################################################################
    coeff_aa_b_i = (
            (6 * ke_b_i ** 2 * de_b_i * R_b_i / (math.pi * E_b_i)) ** (1 / 3)
    )
    aa_b_i = (Q_b_i ** (1 / 3)) * coeff_aa_b_i
    for i in range(n):
        if aa_b_i[i, 0, 0] > 0.9 * R_b:
            aa_b_i[i, 0, 0] = 0.9 * R_b

    bb_b_i = aa_b_i / ke_b_i
    ###########################################################################
    #      Calculate the maximum compressive stress in the contact area       #
    ###########################################################################
    p_b_i_max =np.zeros((n, 1, 1))
    for i in range(n):
        if Q_b_i[i, 0, 0] > 0:
            p_b_i_max[i, 0, 0] = (
                1.5 * Q_b_i[i, 0, 0] /
                (math.pi * aa_b_i[i, 0, 0] * bb_b_i[i, 0, 0])
            )
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    Info_brcf = (delta_b_o,
                 # Contact deflection between ball and outer race.
                 delta_b_i,
                 # Contact deflection between ball and inner race.
                 aa_b_o,
                 # Major width between ball and outer race.
                 aa_b_i,
                 # Major width between ball and inner race.
                 bb_b_o,
                 # Minor width between ball and outer race.
                 bb_b_i,
                 # Minor width between ball and inner race.
                 p_b_o_max,
                 # Contact pressure between ball and outer race.
                 p_b_i_max,
                 # Contact pressure between ball and inner race.
                 q_b_o,
                 # Contact force of per length between ball and outer race.
                 q_b_i,
                 # Contact force of per length between ball and inner race.
                 Q_b_o,
                 # Contact force between ball and outer race.
                 Q_b_i
                 # Contact force between ball and inner race.
                 )

    return Q_b_o, Q_b_i, Info_brcf

###############################################################################
#      Calculate the normal contact load between ball and secondary race      #
###############################################################################
# @njit(fastmath=False)
def ball_race_contact_force_(delta_b_o_, delta_b_i_, Info_brcs_, mod_brcf_):
    """Solve the transformation relationship between ball and race.

    Parameters
    ----------
    delta_b_o_: np.darray
        Strain between the ball and secondary outer race.
    delta_b_i_: np.darray
        Strain between the ball and secondary inner race.
    mod_brcf_: tuple
        Mode data of ball_race_contact_force_.

    Returns
    -------
    Q_b_o_: np.darray
        Contact force between the ball and secondary outer race of
        ball_race_contact_force.
    Q_b_i_: np.darray
        Contact force between the ball and secondary inner race of
        ball_race_contact_force.
    Info_brcf_: tuple
        Information of ball_race_contact_force_.
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    (E_b_i,
     E_b_o,
     K_b_i,
     K_b_o,
     R_b,
     R_b_i,
     R_b_o,
     Shim_thknss_i,
     Shim_thknss_o,
     de_b_i,
     de_b_o,
     ke_b_i,
     ke_b_o,
     n
     ) = mod_brcf_[0::]
    ###########################################################################
    #               Secondary outer race for shim thickness > 0               #
    ###########################################################################
    if Shim_thknss_o > 0:
        num_con_b_o_ = Info_brcs_[-2]
        if num_con_b_o_ == 0:
            aa_b_o_ = np.zeros((n, 1, 1))
            bb_b_o_ = np.zeros((n, 1, 1))
            p_b_o_max_ = np.zeros((n, 1, 1))
            q_b_o_ = np.zeros((n, 1, 1))
            Q_b_o_ = np.zeros((n, 1, 1))
        else:
            ###################################################################
            #                Calculate the normal contact load                #
            #              between ball and secondary outer race              #
            ###################################################################
            Q_b_o_ = K_b_o * delta_b_o_ ** 1.5

            q_b_o_ = np.zeros((n, 1, 1))
            for i in range(n):
                if Q_b_o_[i, 0, 0] > 0:
                    q_b_o_[i, 0, 0] = Q_b_o_[i, 0, 0] / delta_b_o_[i, 0, 0]
            ###################################################################
            #                Calculate the normal contact shape               #
            #              between ball and secondary outer race              #
            ###################################################################
            coeff_aa_b_o_ = (
                (6 * (ke_b_o ** 2) * de_b_o * R_b_o /
                 (math.pi * E_b_o)) ** (1 / 3)
            )
            aa_b_o_ = (Q_b_o_ ** (1 / 3)) * coeff_aa_b_o_
            for i in range(n):
                if aa_b_o_[i, 0, 0] > 0.9 * R_b:
                    aa_b_o_[i, 0, 0] = 0.9 * R_b

            bb_b_o_ = aa_b_o_ / ke_b_o
            ###################################################################
            #  Calculate the maximum compressive stress in the contact area   #
            ###################################################################
            p_b_o_max_ = np.zeros((n, 1, 1))
            for i in range(n):
                if Q_b_o_[i, 0, 0] > 0:
                    p_b_o_max_[i, 0, 0] = (
                        1.5 * Q_b_o_[i, 0, 0] /
                        (math.pi * aa_b_o_[i, 0, 0] * bb_b_o_[i, 0, 0])
                    )
    else:
        aa_b_o_ = np.zeros((n, 1, 1))
        bb_b_o_ = np.zeros((n, 1, 1))
        p_b_o_max_ = np.zeros((n, 1, 1))
        q_b_o_ = np.zeros((n, 1, 1))
        Q_b_o_ = np.zeros((n, 1, 1))
    ###########################################################################
    #               Secondary inner race for shim thickness > 0               #
    ###########################################################################
    if Shim_thknss_i > 0:
        num_con_b_i_ = Info_brcs_[-1]
        if num_con_b_i_ == 0:
            aa_b_i_ = np.zeros((n, 1, 1))
            bb_b_i_ = np.zeros((n, 1, 1))
            p_b_i_max_ = np.zeros((n, 1, 1))
            q_b_i_ = np.zeros((n, 1, 1))
            Q_b_i_ = np.zeros((n, 1, 1))
        else:
            ###################################################################
            #                Calculate the normal contact load                #
            #              between ball and secondary inner race              #
            ###################################################################
            Q_b_i_ = K_b_i * delta_b_i_ ** 1.5

            q_b_i_ = np.zeros((n, 1, 1))
            for i in range(n):
                if Q_b_i_[i, 0, 0] > 0:
                    q_b_i_[i, 0, 0] = Q_b_i_[i, 0, 0] / delta_b_i_[i, 0, 0]
            ###################################################################
            #                Calculate the normal contact shape               #
            #              between ball and secondary inner race              #
            ###################################################################
            coeff_aa_b_i_ = (
                (6 * ke_b_i ** 2 * de_b_i * R_b_i /
                 (math.pi * E_b_i)) ** (1 / 3)
            )
            aa_b_i_ = (Q_b_i_ ** (1 / 3)) * coeff_aa_b_i_
            for i in range(n):
                if aa_b_i_[i, 0, 0] > 0.9 * R_b:
                    aa_b_i_[i, 0, 0] = 0.9 * R_b

            bb_b_i_ = aa_b_i_ / ke_b_i
            ###################################################################
            #  Calculate the maximum compressive stress in the contact area   #
            ###################################################################
            p_b_i_max_ = np.zeros((n, 1, 1))
            for i in range(n):
                if Q_b_i_[i, 0, 0] > 0:
                    p_b_i_max_[i, 0, 0] = (
                        1.5 * Q_b_i_[i, 0, 0] /
                        (math.pi * aa_b_i_[i, 0, 0] * bb_b_i_[i, 0, 0])
                    )
    else:
        aa_b_i_ = np.zeros((n, 1, 1))
        bb_b_i_ = np.zeros((n, 1, 1))
        p_b_i_max_ = np.zeros((n, 1, 1))
        q_b_i_ = np.zeros((n, 1, 1))
        Q_b_i_ = np.zeros((n, 1, 1))
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    Info_brcf_ = (delta_b_o_,
                  # Contact deflection between ball and secondary outer race.
                  delta_b_i_,
                  # Contact deflection between ball and secondary inner race.
                  aa_b_o_,
                  # Major width between ball and secondary outer race.
                  aa_b_i_,
                  # Major width between ball and secondary inner race.
                  bb_b_o_,
                  # Minor width between ball and secondary outer race.
                  bb_b_i_,
                  # Minor width between ball and secondary inner race.
                  p_b_o_max_,
                  # Contact pressure between ball and secondary outer race.
                  p_b_i_max_,
                  # Contact pressure between ball and secondary inner race.
                  q_b_o_,
                  # Contact force of per length between ball and secondary
                  # outer race.
                  q_b_i_,
                  # Contact force of per length between ball and secondary
                  # inner race.
                  Q_b_o_,
                  # Contact force between ball and secondary outer race.
                  Q_b_i_
                  # Contact force between ball and secondary inner race.
                  )

    return Q_b_o_, Q_b_i_, Info_brcf_

###############################################################################
#                      Calculate ball centrifugal forece                      #
###############################################################################
# @njit(fastmath=False)
def ball_centrifugal_forece(x, Info_brcs, Info_brcf, mod_cf):
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    (D_b,
     I_b_z,
     R_b,
     ee_b_o,
     ee_b_i,
     f_i,
     f_o,
     m_b,
     n,
     rpm_i,
     rpm_o) = mod_cf[0::]
    ###########################################################################
    #                               End prepare                               #
    ###########################################################################
    r_og_I, r_ig_I, r_bg_I = Info_brcs[11], Info_brcs[13], Info_brcs[15]
    T_I_a, T_a_oc, T_a_ic = Info_brcs[5], Info_brcs[8], Info_brcs[9]
    alpha_b_o_0, alpha_b_i_0 = Info_brcs[20], Info_brcs[22]
    aa_b_o, aa_b_i = Info_brcf[2], Info_brcf[3]
    Q_b_o, Q_b_i = Info_brcf[10], Info_brcf[11]

    Pr_sur_rad_o = 2 * f_o * D_b / (2 * f_o + 1)
    Pr_sur_rad_i = 2 * f_i * D_b / (2 * f_i + 1)

    T_I_oc = np.zeros((n, 3, 3))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 3))
    for i in range(n):
        t0[:, :] = T_a_oc[i, :, :]
        t1[:, :] = T_I_a[i, :, :]
        t2 = np.dot(t0, t1)
        T_I_oc[i, :, :] = t2

    r_bg_og_I = r_bg_I - r_og_I

    r_os_bg_oc = np.zeros((n, 3, 1))
    r_os_bg_oc[:, 2, 0] = R_b

    r_bg_og_oc = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 1))
    for i in range(n):
        t0[:, :] = T_I_oc[i, :, :]
        t1[:, :] = r_bg_og_I[i, :, :]
        t2 = np.dot(t0, t1)
        r_bg_og_oc[i, :, :] = t2

    r_os_og_oc = r_os_bg_oc + r_bg_og_oc

    T_I_ic = np.zeros((n, 3, 3))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 3))
    for i in range(n):
        t0[:, :] = T_a_ic[i, :, :]
        t1[:, :] = T_I_a[i, :, :]
        t2 = np.dot(t0, t1)
        T_I_ic[i, :, :] = t2

    r_bg_ig_I = r_bg_I - r_ig_I

    r_is_bg_ic = np.zeros((n, 3, 1))
    r_is_bg_ic[:, 2, 0] = R_b

    r_bg_ig_ic = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 1))
    for i in range(n):
        t0[:, :] = T_I_ic[i, :, :]
        t1[:, :] = r_bg_ig_I[i, :, :]
        t2 = np.dot(t0, t1)
        r_bg_ig_ic[i, :, :] = t2

    r_is_ig_ic = r_is_bg_ic + r_bg_ig_ic

    ae_b_o = aa_b_o * ee_b_o
    cond_o = Q_b_o * ae_b_o * np.cos(math.pi + alpha_b_i_0 - alpha_b_o_0)

    ae_b_i = aa_b_i * ee_b_i
    cond_i = Q_b_i * ae_b_i

    abs_alpha_b_i_0 = math.pi + alpha_b_i_0

    (
        a_I, a_J, o_I, o_J,
        c2_I, cb0_I, cb0_J,
        cr0_I, cr0_J, cb2_I,
        cb2_J, cr2_I, cr2_J
    ) = (
        np.zeros((n, 1, 1)), np.zeros((n, 1, 1)), np.zeros((n, 1, 1)),
        np.zeros((n, 1, 1)), np.zeros((n, 1, 1)), np.zeros((n, 1, 1)),
        np.zeros((n, 1, 1)), np.zeros((n, 1, 1)), np.zeros((n, 1, 1)),
        np.zeros((n, 1, 1)), np.zeros((n, 1, 1)), np.zeros((n, 1, 1)),
        np.zeros((n, 1, 1))
        )
    for i in range(n):
        if cond_o[i, 0, 0] >= cond_i[i, 0, 0]:
            a_I[i, 0, 0] = alpha_b_o_0[i, 0, 0]
            a_J[i, 0, 0] = abs_alpha_b_i_0[i, 0, 0]

            o_I[i, 0, 0] = rpm_o * math.pi / 30
            o_J[i, 0, 0] = rpm_i * math.pi / 30

            c2_I[i, 0, 0] = ((
                Pr_sur_rad_o ** 2 -
                (1 * 0.34729636 * aa_b_o[i, 0, 0]) ** 2
                ) ** 0.5 -
                Pr_sur_rad_o
            )

            cb0_I[i, 0, 0] = r_os_bg_oc[i, 0, 0]
            cb0_J[i, 0, 0] = r_is_bg_ic[i, 0, 0]
            cr0_I[i, 0, 0] = r_os_og_oc[i, 0, 0]
            cr0_J[i, 0, 0] = r_is_ig_ic[i, 0, 0]
            cb2_I[i, 0, 0] = r_os_bg_oc[i, 2, 0]
            cb2_J[i, 0, 0] = r_is_bg_ic[i, 2, 0]
            cr2_I[i, 0, 0] = r_os_og_oc[i, 2, 0]
            cr2_J[i, 0, 0] = r_is_ig_ic[i, 2, 0]
        else:
            a_I[i, 0, 0] = abs_alpha_b_i_0[i, 0, 0]
            a_J[i, 0, 0] = alpha_b_o_0[i, 0, 0]

            o_I[i, 0, 0] = rpm_i * math.pi / 30
            o_J[i, 0, 0] = rpm_o * math.pi / 30

            c2_I[i, 0, 0] = ((
                Pr_sur_rad_i ** 2 -
                (0 * 0.34729636 * aa_b_i[i, 0, 0]) ** 2
                ) ** 0.5 -
                Pr_sur_rad_i
            )

            cb0_I[i, 0, 0] = r_is_bg_ic[i, 0, 0]
            cb0_J[i, 0, 0] = r_os_bg_oc[i, 0, 0]
            cr0_I[i, 0, 0] = r_is_ig_ic[i, 0, 0]
            cr0_J[i, 0, 0] = r_os_og_oc[i, 0, 0]
            cb2_I[i, 0, 0] = r_is_bg_ic[i, 2, 0]
            cb2_J[i, 0, 0] = r_os_bg_oc[i, 2, 0]
            cr2_I[i, 0, 0] = r_is_ig_ic[i, 2, 0]
            cr2_J[i, 0, 0] = r_os_og_oc[i, 2, 0]

    X = (cb0_I * np.sin(a_I) -
         cr0_I * np.sin(a_I) +
         (cr2_I + c2_I) * np.cos(a_I))
    Y = -cr0_J * np.sin(a_J) + cr2_J * np.cos(a_J)

    A = np.zeros((n, 3, 3))
    A[:, 0, 0] = np.sin(a_I[:, 0, 0])
    A[:, 0, 1] = np.cos(a_I[:, 0, 0])
    A[:, 0, 2] = np.sin(a_I[:, 0, 0])
    A[:, 1, 0] = (cb2_I[:, 0, 0] + c2_I[:, 0, 0]) * np.cos(a_I[:, 0, 0])
    A[:, 1, 1] = -(cb2_I[:, 0, 0] + c2_I[:, 0, 0]) * np.sin(a_I[:, 0, 0])
    A[:, 1, 2] = X[:, 0, 0]
    A[:, 2, 0] = (-cb0_J[:, 0, 0] * np.sin(a_J[:, 0, 0]) +
                  cb2_J[:, 0, 0] * np.cos(a_J[:, 0, 0]))
    A[:, 2, 1] = (-cb0_J[:, 0, 0] * np.cos(a_J[:, 0, 0]) -
                  cb2_J[:, 0, 0] * np.sin(a_J[:, 0, 0]))
    A[:, 2, 2] = Y[:, 0, 0]

    B = np.zeros((n, 3, 1))
    B[:, 0, 0] = o_I[:, 0, 0] * np.sin(a_I[:, 0, 0])
    B[:, 1, 0] = o_I[:, 0, 0] * X[:, 0, 0]
    B[:, 2, 0] = o_J[:, 0, 0] * Y[:, 0, 0]

    C = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 1))
    for i in range(n):
        t0[:, :] = A[i, :, :]
        t1[:, :] = B[i, :, :]
        t2 = np.linalg.solve(t0, t1)
        C[i, :, :] = t2

    F_b_cff = np.zeros((n, 1, 1))
    F_b_cff[:, 0, 0] = m_b * x[26:24+12*n:12] * C[:, 2, 0] ** 2

    G_b_gyro = np.zeros((n, 1, 1))
    G_b_gyro[:, 0, 0] = -I_b_z * C[:, 1, 0] * C[:, 2, 0]

    ic = np.zeros((n, 1, 1))
    for i in range(n):
        if cond_o[i, 0, 0] <= cond_i[i, 0, 0]:
            ic[i, 0, 0] = 1

    ixlm = 1 - 0.5 * ic

    F_b_gyro = np.zeros((n, 1, 2))
    F_b_gyro[:, 0, 0] = ixlm[:, 0, 0] * R_b * G_b_gyro[:, 0, 0]
    F_b_gyro[:, 0, 1] = (1 - ixlm[:, 0, 0]) * R_b * G_b_gyro[:, 0, 0]
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    Info_cf = (C,
               # Orbit and spin angular velocitiy.
               F_b_cff,
               # Centrifugal forece of ball.
               F_b_gyro,
               # Gyroscope force of ball.
               G_b_gyro,
               # Gyroscope moment of ball.
               )

    return Info_cf

###############################################################################
#                Calculate the traction force of ball and race                #
###############################################################################
# @njit(fastmath=False)
def ball_race_traction_force(x, Info_tc, Info_brcs, Info_brcf, mod_brtf):
    """Solve the ball and cage force.

    Parameters
    ----------
    x: np.darray
        Solution vector.
    Info_tc: tuple
        Information of temperature_change.
    Info_brcs: tuple
        Information of ball_race_contact_strain.
    Info_brcf: tuple
        Information data of ball_race_contact_force.
    mod_brtf: tuple
        Mode data of ball_race_traction_force.

    Returns
    -------
    F_b_r: tuple
        Resultant force between the ball and race.
    M_b_r: tuple
        Resultant moment between the ball and race.
    Info_brtf: tuple
        Information of ball_race_traction_force.
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    (A_0,
     B_0,
     C_0,
     D_0,
     D_b,
     R_b,
     R_yipu_b_o,
     R_yipu_b_i,
     b_i_limt_film,
     b_o_limt_film,
     b_r_lub_type,
     dmpg_b_i,
     dmpg_b_o,
     ep_b_i,
     ep_b_o,
     f_i,
     f_o,
     hj,
     k_b_r_trac_type,
     m_b,
     m_i,
     m_o,
     n,
     oil_type,
     r_bg_bm_b,
     str_parm,
     tj
     ) = mod_brtf[0::]
    ###########################################################################
    #                               End prepare                               #
    ###########################################################################
    temp_o, temp_i = Info_tc[0:2]
    (T_o_I, T_I_o, T_i_I, T_I_i, T_a_I, T_I_a, T_b_I, T_I_b, T_a_oc, T_a_ic,
     r_om_I, r_og_I, r_im_I, r_ig_I, r_bm_I, r_bg_I) = Info_brcs[0:16]
    (delta_b_o, delta_b_i, aa_b_o, aa_b_i, bb_b_o, bb_b_i,
     p_b_o_max, p_b_i_max, q_b_o, q_b_i, Q_b_o, Q_b_i) = Info_brcf[0:12]

    Pr_sur_rad_o = 2 * f_o * D_b / (2 * f_o + 1)
    Pr_sur_rad_i = 2 * f_i * D_b / (2 * f_i + 1)

    T_oc_a = np.transpose(T_a_oc, (0,2,1))

    T_ic_a = np.transpose(T_a_ic, (0,2,1))

    T_I_oc = np.zeros((n, 3, 3))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 3))

    for i in range(n):
        t0[:, :] = T_a_oc[i, :, :]
        t1[:, :] = T_I_a[i, :, :]
        t2 = np.dot(t0, t1)
        T_I_oc[i, :, :] = t2

    T_oc_I = np.transpose(T_I_oc, (0,2,1))

    T_I_ic = np.zeros((n, 3, 3))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 3))

    for i in range(n):
        t0[:, :] = T_a_ic[i, :, :]
        t1[:, :] = T_I_a[i, :, :]
        t2 = np.dot(t0, t1)
        T_I_ic[i, :, :] = t2

    T_ic_I = np.transpose(T_I_ic, (0,2,1))

    T_oc_o = np.zeros((n, 3, 3))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 3))

    for i in range(n):
        t0[:, :] = T_I_o[0, :, :]
        t1[:, :] = T_oc_I[i, :, :]
        t2 = np.dot(t0, t1)
        T_oc_o[i, :, :] = t2

    T_ic_i = np.zeros((n, 3, 3))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 3))

    for i in range(n):
        t0[:, :] = T_I_i[0, :, :]
        t1[:, :] = T_ic_I[i, :, :]
        t2 = np.dot(t0, t1)
        T_ic_i[i, :, :] = t2

    T_b_oc = np.zeros((n, 3, 3))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 3))

    for i in range(n):
        t0[:, :] = T_I_oc[i, :, :]
        t1[:, :] = T_b_I[i, :, :]
        t2 = np.dot(t0, t1)
        T_b_oc[i, :, :] = t2

    T_oc_b = np.transpose(T_b_oc, (0,2,1))

    T_b_ic = np.zeros((n, 3, 3))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 3))

    for i in range(n):
        t0[:, :] = T_I_ic[i, :, :]
        t1[:, :] = T_b_I[i, :, :]
        t2 = np.dot(t0, t1)
        T_b_ic[i, :, :] = t2

    T_ic_b = np.transpose(T_b_ic, (0,2,1))

    v_b_a = np.zeros((n, 3, 1))
    v_b_a[:, 0, 0] = x[25:24+12*n:12]
    v_b_a[:, 2, 0] = x[27:24+12*n:12]

    sin_x32 = np.sin(x[32:24+12*n:12])
    cos_x32 = np.cos(x[32:24+12*n:12])
    sin_x34 = np.sin(x[34:24+12*n:12])
    cos_x34 = np.cos(x[34:24+12*n:12])

    x31 = x[31:24+12*n:12]
    x33 = x[33:24+12*n:12]
    x35 = x[35:24+12*n:12]

    omega_b_b = np.zeros((n, 3, 1))
    omega_b_b[:, 0, 0] = cos_x32 * cos_x34 * x31 + sin_x34 * x33
    omega_b_b[:, 1, 0] = -cos_x32 * sin_x34 * x31 + cos_x34 * x33
    omega_b_b[:, 2, 0] = sin_x32 * x31 + x35

    v_o_I = np.array([[[x[25+12*n]], [x[27+12*n]], [x[29+12*n]]]])

    sin_x32_n = math.sin(x[32 + 12 * n])
    cos_x32_n = math.cos(x[32 + 12 * n])
    sin_x34_n = math.sin(x[34 + 12 * n])
    cos_x34_n = math.cos(x[34 + 12 * n])

    x31_n = x[31 + 12 * n]
    x33_n = x[33 + 12 * n]
    x35_n = x[35 + 12 * n]

    omega_o_o = np.zeros((1, 3, 1))
    omega_o_o[0, 0, 0] = cos_x32_n * cos_x34_n * x31_n + sin_x34_n * x33_n
    omega_o_o[0, 1, 0] = -cos_x32_n * sin_x34_n * x31_n + cos_x34_n * x33_n
    omega_o_o[0, 2, 0] = sin_x32_n * x31_n + x35_n

    v_i_I = np.array([[[x[1]], [x[3]], [x[5]]]])

    sin_x8 = math.sin(x[8])
    cos_x8 = math.cos(x[8])
    sin_x10 = math.sin(x[10])
    cos_x10 = math.cos(x[10])

    x7, x9, x11 = x[7], x[9], x[11]

    omega_i_i = np.zeros((1, 3, 1))
    omega_i_i[0, 0, 0] = cos_x8 * cos_x10 * x7 + sin_x10 * x9
    omega_i_i[0, 1, 0] = -cos_x8 * sin_x10 * x7 + cos_x10 * x9
    omega_i_i[0, 2, 0] = sin_x8 * x7 + x11

    u_r_oc_0 = np.zeros((n, 3, 1))
    u_r_oc_0[:, 0, 0] = x[29:24+12*n:12]

    u_r_ic_0 = np.zeros((n, 3, 1))
    u_r_ic_0[:, 0, 0] = x[29:24+12*n:12]

    v_b_oc = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 1))

    for i in range(n):
        t0[:, :] = T_a_oc[i, :, :]
        t1[:, :] = v_b_a[i, :, :]
        t2 = np.dot(t0, t1)
        v_b_oc[i, :, :] = t2

    v_b_ic = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 1))

    for i in range(n):
        t0[:, :] = T_a_ic[i, :, :]
        t1[:, :] = v_b_a[i, :, :]
        t2 = np.dot(t0, t1)
        v_b_ic[i, :, :] = t2

    omega_b_oc = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 1))

    for i in range(n):
        t0[:, :] = T_b_oc[i, :, :]
        t1[:, :] = omega_b_b[i, :, :]
        t2 = np.dot(t0, t1)
        omega_b_oc[i, :, :] = t2

    omega_b_ic = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 1))

    for i in range(n):
        t0[:, :] = T_b_ic[i, :, :]
        t1[:, :] = omega_b_b[i, :, :]
        t2 = np.dot(t0, t1)
        omega_b_ic[i, :, :] = t2

    v_o_oc = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 1))

    for i in range(n):
        t0[:, :] = T_I_oc[i, :, :]
        t1[:, :] = v_o_I[0, :, :]
        t2 = np.dot(t0, t1)
        v_o_oc[i, :, :] = t2

    v_i_ic = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 1))

    for i in range(n):
        t0[:, :] = T_I_ic[i, :, :]
        t1[:, :] = v_i_I[0, :, :]
        t2 = np.dot(t0, t1)
        v_i_ic[i, :, :] = t2

    omega_o_oc = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 3))
    t2 = np.zeros((3, 1))
    t3 = np.zeros((3, 1))

    for i in range(n):
        t0[:, :] = T_I_oc[i, :, :]
        t1[:, :] = T_o_I[0, :, :]
        t2[:, :] = omega_o_o[0, :, :]
        t3[:, :] = u_r_oc_0[i, :, :]
        t4 = np.dot(t0, (np.dot(t1, t2) - t3))
        omega_o_oc[i, :, :] = t4

    omega_i_ic = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 3))
    t2 = np.zeros((3, 1))
    t3 = np.zeros((3, 1))

    for i in range(n):
        t0[:, :] = T_I_ic[i, :, :]
        t1[:, :] = T_i_I[0, :, :]
        t2[:, :] = omega_i_i[0, :, :]
        t3[:, :] = u_r_ic_0[i, :, :]
        t4 = np.dot(t0, (np.dot(t1, t2) - t3))
        omega_i_ic[i, :, :] = t4

    r_bm_om_oc = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 1))

    for i in range(n):
        t0[:, :] = T_I_oc[i, :, :]
        t1[:, :] = (r_bm_I - r_om_I)[i, :, :]
        t2 = np.dot(t0, t1)
        r_bm_om_oc[i, :, :] = t2

    r_bm_im_ic = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 1))

    for i in range(n):
        t0[:, :] = T_I_ic[i, :, :]
        t1[:, :] = (r_bm_I - r_im_I)[i, :, :]
        t2 = np.dot(t0, t1)
        r_bm_im_ic[i, :, :] = t2

    r_bg_bm_oc = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 1))

    for i in range(n):
        t0[:, :] = T_b_oc[i, :, :]
        t1[: ,:] = r_bg_bm_b[i, :, :]
        t2 = np.dot(t0, t1)
        r_bg_bm_oc[i, :, :] = t2

    r_bg_bm_ic = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 1))

    for i in range(n):
        t0[:, :] = T_b_ic[i, :, :]
        t1[:, :] = r_bg_bm_b[i, :, :]
        t2 = np.dot(t0, t1)
        r_bg_bm_ic[i, :, :] = t2
    ###########################################################################
    #          Calculate the slip zero point of ball and outer race           #
    ###########################################################################
    zero_point_o_l, zero_point_o_r = np.zeros((n, 1, 1)), np.zeros((n, 1, 1))
    for j in range(n):
        if aa_b_o[j, 0, 0] > 0:
            common_params = (
                Pr_sur_rad_o, R_b, aa_b_o[j, 0, 0],
                v_o_oc[j, 1, 0], v_b_oc[j, 1, 0],
                omega_o_oc[j, 0, 0], omega_o_oc[j, 2, 0],
                r_bm_om_oc[j, 0, 0], r_bm_om_oc[j, 2, 0],
                omega_b_oc[j, 0, 0], omega_b_oc[j, 2, 0],
                r_bg_bm_oc[j, 0, 0], r_bg_bm_oc[j, 2, 0]
            )
            zero_point__1_o = slip_zero(-1., *common_params)
            zero_point_0_o = slip_zero(0., *common_params)
            zero_point_1_o = slip_zero(1., *common_params)
            if zero_point__1_o * zero_point_0_o < 0:
                zero_point_o_l[j, 0, 0] = solve_slip_zero(
                    -1., 0., *common_params
                )
            else:
                zero_point_o_l[j, 0, 0] = 0.
            if zero_point_0_o*zero_point_1_o <= 0:
                zero_point_o_r[j, 0, 0] = solve_slip_zero(
                    0., 1., *common_params
                )
            else:
                zero_point_o_r[j, 0, 0] = 0.
    ###########################################################################
    #          Calculate the slip zero point of ball and inner race           #
    ###########################################################################
    zero_point_i_l, zero_point_i_r = np.zeros((n, 1, 1)), np.zeros((n, 1, 1))
    for j in range(n):
        if aa_b_i[j, 0, 0] > 0:
            common_params = (
                Pr_sur_rad_i, R_b, aa_b_i[j, 0, 0],
                v_i_ic[j, 1, 0], v_b_ic[j, 1, 0],
                omega_i_ic[j, 0, 0], omega_i_ic[j, 2, 0],
                r_bm_im_ic[j, 0, 0], r_bm_im_ic[j, 2, 0],
                omega_b_ic[j, 0, 0], omega_b_ic[j, 2, 0],
                r_bg_bm_ic[j, 0, 0], r_bg_bm_ic[j, 2, 0]
            )
            zero_point__1_i = slip_zero(-1., *common_params)
            zero_point_0_i = slip_zero(0., *common_params)
            zero_point_1_i = slip_zero(1., *common_params)
            if zero_point__1_i * zero_point_0_i < 0:
                zero_point_i_l[j, 0, 0] = solve_slip_zero(
                    -1., 0., *common_params
                )
            else:
                zero_point_i_l[j, 0, 0] = 0.
            if zero_point_0_i * zero_point_1_i <= 0:
                zero_point_i_r[j, 0, 0] = solve_slip_zero(
                    0., 1., *common_params
                )
            else:
                zero_point_i_r[j, 0, 0] = 0.
    ###########################################################################
    #                             Single integral                             #
    ###########################################################################
    if b_r_lub_type == 0:
        #######################################################################
        #         Calculate the slip behavior of ball and outer race          #
        #######################################################################
        kkk0, kkk1, kkk2 = value_2(zero_point_o_l, zero_point_o_r, tj)

        kkk = np.zeros((n, 1, 12))
        kkk[:, :, 0:4], kkk[:, :, 4:8], kkk[:, :, 8:12] = kkk0, kkk1, kkk2

        hhh = np.zeros((n, 1, 12))
        hhh[:, :, 0:4] = (zero_point_o_l + 1) * 0.5 * hj
        hhh[:, :, 4:8] = (zero_point_o_r - zero_point_o_l) * 0.5 * hj
        hhh[:, :, 8:12] = (1 - zero_point_o_r) * 0.5 * hj

        thetak_b_o = np.arcsin(kkk * aa_b_o / Pr_sur_rad_o)

        r_os_bg_oc_x = kkk * aa_b_o
        r_os_bg_oc_y = 0 * kkk * aa_b_o
        r_os_bg_oc_z = R_b + Pr_sur_rad_o * (np.cos(thetak_b_o) - 1)

        r_os_bg_oc = np.zeros((n, 3, 12))
        r_os_bg_oc[:, 0, :] = r_os_bg_oc_x[:, 0, :]
        r_os_bg_oc[:, 1, :] = r_os_bg_oc_y[:, 0, :]
        r_os_bg_oc[:, 2, :] = r_os_bg_oc_z[:, 0, :]

        r_os_bm_oc = r_os_bg_oc + r_bg_bm_oc
        r_os_om_oc = r_os_bm_oc + r_bm_om_oc
        #######################################################################
        #  Calculate the relative sliding speed between ball and outer race   #
        #######################################################################
        u_r_oc = v_o_oc + value_1(omega_o_oc, r_os_om_oc)
        u_b_oc = v_b_oc + value_1(omega_b_oc, r_os_bm_oc)

        u_s_oc = u_r_oc - u_b_oc

        sin_theta = np.sin(thetak_b_o)
        cos_theta = np.cos(thetak_b_o)

        u_s_ol = np.zeros((n, 3, 12))
        u_s_ol[:, 0, :] = (u_s_oc[:, 0, :] * cos_theta[:, 0, :] -
                           u_s_oc[:, 2, :] * sin_theta[:, 0, :])
        u_s_ol[:, 1, :] = u_s_oc[:, 1, :]
        u_s_ol[:, 2, :] = (u_s_oc[:, 0, :] * sin_theta[:, 0, :] +
                           u_s_oc[:, 2, :] * cos_theta[:, 0, :])

        u_s_ol_norm = np.hypot(u_s_ol[:, 0:1, :], u_s_ol[:, 1:2, :])
        #######################################################################
        #          Calculate the oil film thickness between ball and          #
        #                          outer race center                          #
        #######################################################################
        r_os_bg_oc_ctr = np.zeros((n, 3, 1))
        r_os_bg_oc_ctr[:, 2, 0] = R_b

        r_os_bm_oc_ctr = r_os_bg_oc_ctr + r_bg_bm_oc

        r_os_om_oc_ctr = r_os_bm_oc_ctr + r_bm_om_oc
        r_os_og_oc_ctr = r_os_om_oc_ctr# - r_og_om_oc

        u_r_oc_ctr = v_o_oc + value_1(omega_o_oc, r_os_om_oc_ctr)
        u_b_oc_ctr = v_b_oc + value_1(omega_b_oc, r_os_bm_oc_ctr)

        u_s_oc_ctr = np.abs(u_r_oc_ctr - u_b_oc_ctr)

        u_s_oc_ctr_y = np.zeros((n, 1, 1))
        u_s_oc_ctr_y[:, 0, 0] = u_s_oc_ctr[:, 1, 0]

        spin_roll_b_oc = np.zeros((n, 1, 1))
        for i in range(n):
            if omega_o_oc[i, 0, 0] - omega_b_oc[i, 0, 0] != 0:
                spin_roll_b_oc[i, 0, 0] = (
                    (omega_o_oc[i, 2, 0] - omega_b_oc[i, 2, 0]) /
                    (omega_o_oc[i, 0, 0] - omega_b_oc[i, 0, 0])
                )

        ur_b_oc = np.zeros((n, 1, 1))
        ur_b_oc[:, 0, 0] = np.abs(
            (-omega_b_oc[:, 0, 0] * r_os_bm_oc_ctr[:, 2, 0] -
             omega_o_oc[:, 0, 0] * r_os_om_oc_ctr[:, 2, 0] +
             omega_o_oc[:, 2, 0] * r_os_om_oc_ctr[:, 0, 0]) * 0.5
        )

        sli_roll_b_oc = np.zeros((n, 1, 12))
        sli_roll_b_oc[:, 0, :] = u_s_oc[:, 1, :] / ur_b_oc[:, 0, :]

        oil_prop_b_o = oil_main(oil_type, temp_o, 0)
        vis_lub_b_o = oil_prop_b_o[3]
        vis_coeff_0_b_o = oil_prop_b_o[7]
        dvis_lub_b_o = oil_prop_b_o[9]
        ther_cond_lub_b_o = oil_prop_b_o[6]

        common_params = (
            R_yipu_b_o, ep_b_o, vis_lub_b_o, vis_coeff_0_b_o,
            dvis_lub_b_o, ther_cond_lub_b_o, str_parm, 0
        )

        h_iso_b_o, phit_b_o, sip_b_o = (
            np.zeros((n ,1, 1)), np.zeros((n ,1, 1)), np.zeros((n ,1, 1))
        )
        for i in range(n):
            (
                p_b_o_max_ij, aa_b_o_ij, bb_b_o_ij,
                ur_b_oc_ij, u_s_oc_ctr_y_ij
                ) = (
                    p_b_o_max[i, 0, 0], aa_b_o[i, 0, 0], bb_b_o[i, 0, 0],
                    ur_b_oc[i, 0, 0], u_s_oc_ctr_y[i, 0, 0]
                    )
            h_iso_b_o_ij, phit_b_o_ij, sip_b_o_ij = film_thickness(
                p_b_o_max_ij, aa_b_o_ij, bb_b_o_ij,
                ur_b_oc_ij, u_s_oc_ctr_y_ij,
                *common_params
            )
            h_iso_b_o[i, 0, 0], phit_b_o[i, 0, 0], sip_b_o[i, 0, 0] = (
                h_iso_b_o_ij, phit_b_o_ij, sip_b_o_ij
            )

        h_ts_b_o = h_iso_b_o * phit_b_o

        r_os_bg_b_ctr = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_oc_b[i, :, :]
            t1[:, :] = r_os_bg_oc_ctr[i, :, :]
            t2 = np.dot(t0, t1)
            r_os_bg_b_ctr[i, :, :] = t2

        r_os_og_o_ctr = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_oc_o[i, :, :]
            t1[:, :] = r_os_og_oc_ctr[i, :, :]
            t2 = np.dot(t0, t1)
            r_os_og_o_ctr[i, :, :] = t2
        #######################################################################
        #                   Simple A, B, C, D traction model                  #
        #######################################################################
        if k_b_r_trac_type == 0:
            u_s_ol_norm_abs = np.zeros((n, 1, 12))
            u_s_ol_norm_abs[:, 0, :] = np.abs(u_s_ol_norm[:, 0, :])

            exp_term = np.exp(-C_0 * u_s_ol_norm_abs)

            miu_o_ol_norm = np.zeros((n, 1, 12))
            miu_o_ol_norm = (A_0 + B_0 * u_s_ol_norm_abs) * exp_term + D_0

            miu_o_ol = np.zeros((n, 3, 12))
            for i in range(n):
                for j in range(12):
                    if u_s_ol_norm[i, 0, j] != 0:
                        miu_o_ol[i, 0:2, j] = (
                            u_s_ol[i, 0:2, j] / u_s_ol_norm[i, 0, j] *
                            miu_o_ol_norm[i, 0, j]
                        )

            miu_o_oc = np.zeros((n, 3, 12))
            miu_o_oc[:, 0:2, :] = miu_o_ol[:, 0:2, :]
        #######################################################################
        #                     User-defined traction model                     #
        #######################################################################
        elif k_b_r_trac_type == -1:
            pass
            """
            ktc = np.zeros_like(h_ts_b_o)
            ktc[h_ts_b_o <= b_o_limt_film] = 1

            miu_o_ol_norm = sub_traction_coefficient(
                ktc, u_s_ol_norm, ur_b_oc, p_b_o_max,
                aa_b_o, bb_b_o, h_ts_b_o, temp_o, 0
            )

            miu_o_ol = np.zeros((n, 3, 12))
            for i in range(n):
                for j in range(12):
                    if u_s_ol_norm[i, 0, j] != 0:
                        miu_o_ol[i, 0:2, j] = (
                            u_s_ol[i, 0:2, j] / u_s_ol_norm[i, 0, j] *
                            miu_o_ol_norm[i, 0, j]
                        )

            miu_o_oc = np.zeros((n, 3, 12))
            miu_o_oc[:, 0:2, :] = miu_o_ol[:, 0:2, :]
            """
        #######################################################################
        #                          Force and moment                           #
        #######################################################################
        k_b_o = np.zeros((n, 1, 1))
        for i in range(n):
            if delta_b_o[i, 0, 0] != 0:
                k_b_o[i, 0, 0] = 1.5 * Q_b_o[i, 0, 0] / delta_b_o[i, 0, 0]

        brdq_b_o = damping_coefficient(
            dmpg_b_o, 0, m_b, m_o, k_b_o, Q_b_o, 1, 0
        )

        df_b_o = math.pi * 0.5 * p_b_o_max * aa_b_o * bb_b_o * (1 - kkk ** 2)

        pv_b_o = p_b_o_max * np.sqrt(1 - kkk ** 2) * u_s_ol_norm
        qvk_b_o = df_b_o * u_s_ol_norm * hhh

        qv_b_o = np.zeros((n, 1, 1))
        qv_b_o[:, 0, :] = np.sum(qvk_b_o, axis=2)

        sv_b_o = np.zeros((n, 1, 1))
        sv_b_o[:, 0, :] = np.sum(qvk_b_o*miu_o_ol_norm, axis=2)

        unit_array_z = np.array([[[0], [0], [1]]])

        Int0 = (df_b_o * (miu_o_oc - unit_array_z) +
                brdq_b_o * u_s_oc[:, 2:3, :] * unit_array_z)
        Int01 = hhh * value_1(r_os_om_oc, -Int0)

        Int02 = np.zeros((n, 3, 12))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            for j in range(12):
                t0[:, :] = T_oc_o[i, :, :]
                t1[:, :] = Int01[i, :, j:j+1]
                t2 = np.dot(t0, t1)
                Int02[i, :, j:j+1] = t2

        F_b_o_oc = np.zeros((n, 3, 1))
        F_b_o_oc[:, :, 0] = np.sum(Int0*hhh, axis=2)

        M_b_o_oc = np.zeros((n, 3, 1))
        M_b_o_oc[:, :, 0] = np.sum(hhh*value_1(r_os_bm_oc, Int0), axis=2)

        F_b_o_a = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_oc_a[i, :, :]
            t1[:, :] = F_b_o_oc[i, :, :]
            t2 = np.dot(t0, t1)
            F_b_o_a[i, :, :] = t2

        F_b_o_I = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_oc_I[i, :, :]
            t1[:, :] = F_b_o_oc[i, :, :]
            t2 = np.dot(t0, t1)
            F_b_o_I[i, :, :] = t2

        F_o_b_I = np.zeros((1, 3, 1))
        F_o_b_I[0,:,:] = np.sum(-F_b_o_I, axis=0)

        M_b_o_b = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_oc_b[i, :, :]
            t1[:, :] = M_b_o_oc[i, :, :]
            t2 = np.dot(t0, t1)
            M_b_o_b[i, :, :] = t2

        M_o_b_o = np.zeros((1, 3, 1))
        M_o_b_o[0, :, 0] = np.sum((np.sum(Int02, axis=2)), axis=0)
        #######################################################################
        #         Calculate the slip behavior of ball and inner race          #
        #######################################################################
        k0, k1, k2 = value_2(zero_point_i_l, zero_point_i_r, tj)

        k = np.zeros((n, 1, 12))
        k[:, :, 0:4], k[:, :, 4:8], k[:, :, 8:12] = k0, k1, k2

        h = np.zeros((n, 1, 12))
        h[:, :, 0:4] = (zero_point_i_l + 1) * 0.5 * hj
        h[:, :, 4:8] = (zero_point_i_r - zero_point_i_l) * 0.5 * hj
        h[:, :, 8:12] = (1 - zero_point_i_r) * 0.5 * hj

        thetak_b_i = np.arcsin(k * aa_b_i / Pr_sur_rad_i)

        r_is_bg_ic_x = k * aa_b_i
        r_is_bg_ic_y = 0 * k * aa_b_i
        r_is_bg_ic_z = R_b + Pr_sur_rad_i * (np.cos(thetak_b_i) - 1)

        r_is_bg_ic = np.zeros((n, 3, 12))
        r_is_bg_ic[:, 0, :] = r_is_bg_ic_x[:, 0, :]
        r_is_bg_ic[:, 1, :] = r_is_bg_ic_y[:, 0, :]
        r_is_bg_ic[:, 2, :] = r_is_bg_ic_z[:, 0, :]

        r_is_bm_ic = r_is_bg_ic + r_bg_bm_ic
        r_is_im_ic = r_is_bm_ic + r_bm_im_ic
        #######################################################################
        #  Calculate the relative sliding speed between ball and inner race   #
        #######################################################################
        u_r_ic = v_i_ic + value_1(omega_i_ic, r_is_im_ic)
        u_b_ic = v_b_ic + value_1(omega_b_ic, r_is_bm_ic)

        u_s_ic = u_r_ic - u_b_ic

        sin_theta = np.sin(thetak_b_i)
        cos_theta = np.cos(thetak_b_i)

        u_s_il = np.zeros((n, 3, 12))
        u_s_il[:, 0, :] = (u_s_ic[:, 0, :] * cos_theta[:, 0, :] -
                           u_s_ic[:, 2, :] * sin_theta[:, 0, :])
        u_s_il[:, 1, :] = u_s_ic[:, 1, :]
        u_s_il[:, 2, :] = (u_s_ic[:, 0, :] * sin_theta[:, 0, :] +
                           u_s_ic[:, 2, :] * cos_theta[:, 0, :])

        u_s_il_norm = np.hypot(u_s_il[:, 0:1, :], u_s_il[:, 1:2, :])
        #######################################################################
        #          Calculate the oil film thickness between ball and          #
        #                          inner race center                          #
        #######################################################################
        r_is_bg_ic_ctr = np.zeros((n, 3, 1))
        r_is_bg_ic_ctr[:, 2, 0] = R_b

        r_is_bm_ic_ctr = r_is_bg_ic_ctr + r_bg_bm_ic

        r_is_im_ic_ctr = r_is_bm_ic_ctr + r_bm_im_ic
        r_is_ig_ic_ctr = r_is_im_ic_ctr# - r_ig_im_ic

        u_r_ic_ctr = v_i_ic + value_1(omega_i_ic, r_is_im_ic_ctr)
        u_b_ic_ctr = v_b_ic + value_1(omega_b_ic, r_is_bm_ic_ctr)

        u_s_ic_ctr = np.abs(u_r_ic_ctr - u_b_ic_ctr)

        u_s_ic_ctr_y = np.zeros((n, 1, 1))
        u_s_ic_ctr_y[:, 0, 0] = u_s_ic_ctr[:, 1, 0]

        spin_roll_b_ic = np.zeros((n, 1, 1))
        for i in range(n):
            if omega_i_ic[i, 0, 0] - omega_b_ic[i, 0, 0] != 0:
                spin_roll_b_ic[i, 0, 0] = (
                    (omega_i_ic[i, 2, 0] - omega_b_ic[i, 2, 0]) /
                    (omega_i_ic[i, 0, 0] - omega_b_ic[i, 0, 0])
                )

        ur_b_ic = np.zeros((n, 1, 1))
        ur_b_ic[:, 0, 0] = np.abs(
            (-omega_b_ic[:, 0, 0] * r_is_bm_ic_ctr[:, 2, 0] -
             omega_i_ic[:, 0, 0] * r_is_im_ic_ctr[:, 2, 0] +
             omega_i_ic[:, 2, 0] * r_is_im_ic_ctr[:, 0, 0]) * 0.5
        )

        sli_roll_b_ic = np.zeros((n,1,12))
        sli_roll_b_ic[:,0,:] = u_s_ic[:,1,:] / ur_b_ic[:,0,:]

        oil_prop_b_i = oil_main(oil_type, temp_i, 0)
        (
            vis_lub_b_i, vis_coeff_0_b_i,
            dvis_lub_b_i, ther_cond_lub_b_i
            ) = (
                oil_prop_b_i[3], oil_prop_b_i[7],
                oil_prop_b_i[9], oil_prop_b_i[6]
                )

        common_params = (
            R_yipu_b_i, ep_b_i, vis_lub_b_i, vis_coeff_0_b_i,
            dvis_lub_b_i, ther_cond_lub_b_i, str_parm, 0
        )

        h_iso_b_i, phit_b_i, sip_b_i = (
            np.zeros((n ,1, 1)), np.zeros((n ,1, 1)), np.zeros((n ,1, 1))
        )
        for i in range(n):
            (
                p_b_i_max_ij, aa_b_i_ij, bb_b_i_ij,
                ur_b_ic_ij, u_s_ic_ctr_y_ij
                ) = (
                    p_b_i_max[i, 0, 0], aa_b_i[i, 0, 0], bb_b_i[i, 0, 0],
                    ur_b_ic[i, 0, 0], u_s_ic_ctr_y[i, 0, 0]
                    )
            h_iso_b_i_ij, phit_b_i_ij, sip_b_i_ij = film_thickness(
                p_b_i_max_ij, aa_b_i_ij, bb_b_i_ij,
                ur_b_ic_ij, u_s_ic_ctr_y_ij,
                *common_params
            )
            h_iso_b_i[i, 0, 0], phit_b_i[i, 0, 0], sip_b_i[i, 0, 0] = (
                h_iso_b_i_ij, phit_b_i_ij, sip_b_i_ij
            )

        h_ts_b_i = h_iso_b_i * phit_b_i

        r_is_bg_b_ctr = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_ic_b[i, :, :]
            t1[:, :] = r_is_bg_ic_ctr[i, :, :]
            t2 = np.dot(t0, t1)
            r_is_bg_b_ctr[i, :, :] = t2

        r_is_ig_i_ctr = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_ic_i[i, :, :]
            t1[:, :] = r_is_ig_ic_ctr[i, :, :]
            t2 = np.dot(t0, t1)
            r_is_ig_i_ctr[i, :, :] = t2
        #######################################################################
        #                   Simple A, B, C, D traction model                  #
        #######################################################################
        if k_b_r_trac_type == 0:
            u_s_il_norm_abs = np.zeros((n, 1, 12))
            u_s_il_norm_abs[:, 0, :] = np.abs(u_s_il_norm[:, 0, :])

            exp_term = np.exp(-C_0 * u_s_il_norm_abs)

            miu_i_il_norm = np.zeros((n, 1, 12))
            miu_i_il_norm = (A_0 + B_0 * u_s_il_norm_abs) * exp_term + D_0

            miu_i_il = np.zeros((n, 3, 12))
            for i in range(n):
                for j in range(12):
                    if u_s_il_norm[i, 0, j] != 0:
                        miu_i_il[i, 0:2, j] = (
                            u_s_il[i, 0:2, j] / u_s_il_norm[i, 0, j] *
                            miu_i_il_norm[i, 0, j]
                        )

            miu_i_ic = np.zeros((n, 3, 12))
            miu_i_ic[:, 0:2, :] = miu_i_il[:, 0:2, :]
        #######################################################################
        #                     User-defined traction model                     #
        #######################################################################
        elif k_b_r_trac_type == -1:
            pass
            """
            ktc = np.zeros_like(h_ts_b_i)
            ktc[h_ts_b_i <= b_i_limt_film] = 1

            miu_i_il_norm = sub_traction_coefficient(
                ktc, u_s_il_norm, ur_b_ic, p_b_i_max,
                aa_b_i, bb_b_i, h_ts_b_i, temp_i, 0
            )

            miu_i_il = np.zeros((n, 3, 12))
            for i in range(n):
                for j in range(12):
                    if u_s_il_norm[i, 0, j] != 0:
                        miu_i_il[i, 0:2, j] = (
                            u_s_il[i, 0:2, j] / u_s_il_norm[i, 0, j] *
                            miu_i_il_norm[i, 0, j]
                        )

            miu_i_ic = np.zeros((n, 3, 12))
            miu_i_ic[:, 0:2, :] = miu_i_il[:, 0:2, :]
            """
        #######################################################################
        #                          Force and moment                           #
        #######################################################################
        k_b_i = np.zeros((n, 1, 1))
        for i in range(n):
            if delta_b_i[i, 0, 0] != 0:
                k_b_i[i, 0, 0] = 1.5 * Q_b_i[i, 0, 0] / delta_b_i[i, 0, 0]

        brdq_b_i = damping_coefficient(
            dmpg_b_i, 0, m_b, m_i, k_b_i, Q_b_i, 1, 0
        )

        df_b_i = math.pi * 0.5 * p_b_i_max * aa_b_i * bb_b_i * (1 - k ** 2)

        pv_b_i = p_b_i_max * np.sqrt(1 - k ** 2) * u_s_il_norm
        qvk_b_i = df_b_i * u_s_il_norm * h

        qv_b_i = np.zeros((n, 1, 1))
        qv_b_i[:, 0, :] = np.sum(qvk_b_i, axis=2)

        sv_b_i = np.zeros((n, 1, 1))
        sv_b_i[:, 0, :] = np.sum(qvk_b_i*miu_i_il_norm, axis=2)

        unit_array_z = np.array([[[0], [0], [1]]])

        Int1 = (df_b_i * (miu_i_ic - unit_array_z) +
                brdq_b_i * u_s_ic[:,2:3,:] * unit_array_z)
        Int11 = h * value_1(r_is_im_ic, -Int1)

        Int12 = np.zeros((n, 3, 12))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            for j in range(12):
                t0[:, :] = T_ic_i[i, :, :]
                t1[:, :] = Int11[i, :, j:j+1]
                t2 = np.dot(t0, t1)
                Int12[i, :, j:j+1] = t2

        F_b_i_ic = np.zeros((n, 3, 1))
        F_b_i_ic[:, :, 0] = np.sum(Int1*h, axis=2)

        M_b_i_ic = np.zeros((n, 3, 1))
        M_b_i_ic[:, :, 0] = np.sum(h*value_1(r_is_bm_ic, Int1), axis=2)

        F_b_i_a = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_ic_a[i, :, :]
            t1[:, :] = F_b_i_ic[i, :, :]
            t2 = np.dot(t0, t1)
            F_b_i_a[i, :, :] = t2

        F_b_i_I = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_ic_I[i, :, :]
            t1[:, :] = F_b_i_ic[i, :, :]
            t2 = np.dot(t0, t1)
            F_b_i_I[i, :, :] = t2

        F_i_b_I = np.zeros((1, 3, 1))
        F_i_b_I[0, :, :] = np.sum(-F_b_i_I, axis=0)

        M_b_i_b = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_ic_b[i, :, :]
            t1[:, :] = M_b_i_ic[i, :, :]
            t2 = np.dot(t0, t1)
            M_b_i_b[i, :, :] = t2

        M_i_b_i = np.zeros((1, 3, 1))
        M_i_b_i[0, :, 0] = np.sum((np.sum(Int12, axis=2)), axis=0)
    ###########################################################################
    #                             Double integral                             #
    ###########################################################################
    else:
        #######################################################################
        #                      Chebyshev integral point                       #
        #######################################################################
        r_j_0, r_j_1, r_j_2, r_j_3, r_j_4 = (
            0.0445594672, 0.218694082, 0.474310986, 0.735889118, 0.929067427
        )
        w_j_0, w_j_1, w_j_2, w_j_3, w_j_4 = (
            0.112314749, 0.220987601, 0.239278553, 0.161581619, 0.0512356414
        )

        m_j = np.zeros((n, 1, 144))
        m_j[:, 0, :] = np.linspace(1, 144, 144)

        c0, c1, c2, c3, c4 = (
            w_j_0 * math.pi * r_j_0 / 72,
            w_j_1 * math.pi * r_j_1 / 72,
            w_j_2 * math.pi * r_j_2 / 72,
            w_j_3 * math.pi * r_j_3 / 72,
            w_j_4 * math.pi * r_j_4 / 72
        )

        sin_ang = np.zeros((n, 1, 720))
        sin_ang[:, 0, :] = np.sin(math.pi * m_j[:, 0, :] / 72)

        cos_ang = np.zeros((n, 1, 720))
        cos_ang[:, 0, :] = np.cos(math.pi * m_j[:, 0, :] / 72)
        #######################################################################
        #         Calculate the slice behavior of ball and outer race         #
        #######################################################################
        r_os_bm_oc_x = np.zeros((n, 1, 720))
        r_os_bm_oc_x[:, 0, 0:144] = r_j_0 * (cos_ang * aa_b_o)[:, 0, :]
        r_os_bm_oc_x[:, 0, 144:288] = r_j_1 * (cos_ang * aa_b_o)[:, 0, :]
        r_os_bm_oc_x[:, 0, 288:432] = r_j_2 * (cos_ang * aa_b_o)[:, 0, :]
        r_os_bm_oc_x[:, 0, 432:576] = r_j_3 * (cos_ang * aa_b_o)[:, 0, :]
        r_os_bm_oc_x[:, 0, 576:720] = r_j_4 * (cos_ang * aa_b_o)[:, 0, :]

        r_os_bm_oc_y = np.zeros((n, 1, 720))
        r_os_bm_oc_y[:, 0, 0:144] = r_j_0 * (sin_ang * bb_b_o)[:, 0, :]
        r_os_bm_oc_y[:, 0, 144:288] = r_j_1 * (sin_ang * bb_b_o)[:, 0, :]
        r_os_bm_oc_y[:, 0, 288:432] = r_j_2 * (sin_ang * bb_b_o)[:, 0, :]
        r_os_bm_oc_y[:, 0, 432:576] = r_j_3 * (sin_ang * bb_b_o)[:, 0, :]
        r_os_bm_oc_y[:, 0, 576:720] = r_j_4 * (sin_ang * bb_b_o)[:, 0, :]

        thetak_b_o = np.arcsin(r_os_bm_oc_x / Pr_sur_rad_o)

        r_os_bm_oc_z = np.zeros((n, 1, 720))
        r_os_bm_oc_z[:, :, 0:144] = (
            R_b + Pr_sur_rad_o * (np.cos(thetak_b_o[:, :, 0:144]) - 1)
        )
        r_os_bm_oc_z[:, :, 144:288] = (
            R_b + Pr_sur_rad_o * (np.cos(thetak_b_o[:, :, 144:288]) - 1)
        )
        r_os_bm_oc_z[:, :, 288:432] = (
            R_b + Pr_sur_rad_o * (np.cos(thetak_b_o[:, :, 288:432]) - 1)
        )
        r_os_bm_oc_z[:, :, 432:576] = (
            R_b + Pr_sur_rad_o * (np.cos(thetak_b_o[:, :, 432:576]) - 1)
        )
        r_os_bm_oc_z[:, :, 576:720] = (
            R_b + Pr_sur_rad_o * (np.cos(thetak_b_o[:, :, 576:720]) - 1)
        )

        r_os_bm_oc = np.zeros((n, 3, 720))
        r_os_bm_oc[:, 0, :] = r_os_bm_oc_x[:, 0, :]
        r_os_bm_oc[:, 1, :] = r_os_bm_oc_y[:, 0, :]
        r_os_bm_oc[:, 2, :] = r_os_bm_oc_z[:, 0, :]

        r_os_om_oc = r_os_bm_oc + r_bm_om_oc
        #######################################################################
        #  Calculate the relative sliding speed between ball and outer race   #
        #######################################################################
        u_r_oc = v_o_oc + value_1(omega_o_oc, r_os_om_oc)
        u_b_oc = v_b_oc + value_1(omega_b_oc, r_os_bm_oc)

        u_s_oc = u_r_oc - u_b_oc
        u_s_oc_norm = np.hypot(u_s_oc[:, 0:1, :] ** 2, u_s_oc[:, 1:2, :] ** 2)
        #######################################################################
        #          Calculate the oil film thickness between ball and          #
        #                          outer race center                          #
        #######################################################################
        r_os_bg_oc_ctr = np.zeros((n, 3, 1))
        r_os_bg_oc_ctr[:, 2, 0] = R_b

        r_os_bm_oc_ctr = r_os_bg_oc_ctr + r_bg_bm_oc

        r_os_om_oc_ctr = r_os_bm_oc_ctr + r_bm_om_oc
        r_os_og_oc_ctr = r_os_om_oc_ctr# - r_og_om_oc

        u_r_oc_ctr = v_o_oc + value_1(omega_o_oc, r_os_om_oc_ctr)
        u_b_oc_ctr = v_b_oc + value_1(omega_b_oc, r_os_bm_oc_ctr)

        u_s_oc_ctr = np.abs(u_r_oc_ctr - u_b_oc_ctr)

        u_s_oc_ctr_y = np.zeros((n, 1, 1))
        u_s_oc_ctr_y[:, 0, 0] = u_s_oc_ctr[:, 1, 0]

        spin_roll_b_oc = np.zeros((n, 1, 1))
        for i in range(n):
            if omega_o_oc[i, 0, 0] - omega_b_oc[i, 0, 0] != 0:
                spin_roll_b_oc[i, 0, 0] = (
                    (omega_o_oc[i, 2, 0] - omega_b_oc[i, 2, 0]) /
                    (omega_o_oc[i, 0, 0] - omega_b_oc[i, 0, 0])
                )

        ur_b_oc = np.zeros((n, 1, 1))
        ur_b_oc[:, 0, 0] = np.abs(
            (-omega_b_oc[:, 0, 0] * r_os_bm_oc_ctr[:, 2, 0] -
             omega_o_oc[:, 0, 0] * r_os_om_oc_ctr[:, 2, 0] +
             omega_o_oc[:, 2, 0] * r_os_om_oc_ctr[:, 0, 0]) * 0.5
        )

        sli_roll_b_oc = np.zeros((n, 1, 12))
        sli_roll_b_oc[:, 0, :] = u_s_oc[:, 1, :] / ur_b_oc[:, 0, :]

        oil_prop_b_o = oil_main(oil_type, temp_o, 0)
        vis_lub_b_o = oil_prop_b_o[3]
        vis_coeff_0_b_o = oil_prop_b_o[7]
        dvis_lub_b_o = oil_prop_b_o[9]
        ther_cond_lub_b_o = oil_prop_b_o[6]

        common_params = (
            R_yipu_b_o, ep_b_o, vis_lub_b_o, vis_coeff_0_b_o,
            dvis_lub_b_o, ther_cond_lub_b_o, str_parm, 0
        )

        h_iso_b_o, phit_b_o, sip_b_o = (
            np.zeros((n ,1, 1)), np.zeros((n ,1, 1)), np.zeros((n ,1, 1))
        )
        for i in range(n):
            (
                p_b_o_max_ij, aa_b_o_ij, bb_b_o_ij,
                ur_b_oc_ij, u_s_oc_ctr_y_ij
                ) = (
                    p_b_o_max[i, 0, 0], aa_b_o[i, 0, 0], bb_b_o[i, 0, 0],
                    ur_b_oc[i, 0, 0], u_s_oc_ctr_y[i, 0, 0]
                    )
            h_iso_b_o_ij, phit_b_o_ij, sip_b_o_ij = film_thickness(
                p_b_o_max_ij, aa_b_o_ij, bb_b_o_ij,
                ur_b_oc_ij, u_s_oc_ctr_y_ij,
                *common_params
            )
            h_iso_b_o[i, 0, 0], phit_b_o[i, 0, 0], sip_b_o[i, 0, 0] = (
                h_iso_b_o_ij, phit_b_o_ij, sip_b_o_ij
            )

        h_ts_b_o = h_iso_b_o * phit_b_o

        r_os_bg_b_ctr = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_oc_b[i, :, :]
            t1[:, :] = r_os_bg_oc_ctr[i, :, :]
            t2 = np.dot(t0, t1)
            r_os_bg_b_ctr[i, :, :] = t2

        r_os_og_o_ctr = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_oc_o[i, :, :]
            t1[:, :] = r_os_og_oc_ctr[i, :, :]
            t2 = np.dot(t0, t1)
            r_os_og_o_ctr[i, :, :] = t2
        #######################################################################
        #                   Simple A, B, C, D traction model                  #
        #######################################################################
        if k_b_r_trac_type == 0:
            u_s_oc_norm_abs = np.zeros((n, 1, 720))
            u_s_oc_norm_abs[:, 0, :] = np.abs(u_s_oc_norm[:, 0, :])

            exp_term = np.exp(-C_0 * u_s_oc_norm_abs)

            miu_o_oc_norm = np.zeros((n, 1, 720))
            miu_o_oc_norm[:, 0, :] = (
                (A_0 + B_0 * u_s_oc_norm_abs) * exp_term + D_0
            )

            miu_o_oc = np.zeros((n, 3, 720))
            for i in range(n):
                for j in range(12):
                    if u_s_oc_norm[i, 0, j] != 0:
                        miu_o_oc[i, 0:2, j] = (
                            u_s_oc[i, 0:2, j] / u_s_oc_norm[i, 0, j] *
                            miu_o_oc_norm[i, 0, j]
                        )
        #######################################################################
        #                     User-defined traction model                     #
        #######################################################################
        elif k_b_r_trac_type == -1:
            pass
            """
            ktc = np.zeros_like(h_ts_b_o)
            ktc[h_ts_b_o <= b_o_limt_film] = 1

            miu_o_oc_norm = sub_traction_coefficient(
                ktc, u_s_oc_norm, ur_b_oc, p_b_o_max,
                aa_b_o, bb_b_o, h_ts_b_o, temp_o, 0
            )

            miu_o_oc = np.zeros((n, 3, 720))
            for i in range(n):
                for j in range(12):
                    if u_s_oc_norm[i, 0, j] != 0:
                        miu_o_oc[i, 0:2, j] = (
                            u_s_oc[i, 0:2, j] / u_s_oc_norm[i, 0, j] *
                            miu_o_oc_norm[i, 0, j]
                        )
            """
        #######################################################################
        #                          Force and moment                           #
        #######################################################################
        r_os_bm_oc_r = np.sqrt(1 - r_os_bm_oc_x ** 2 - r_os_bm_oc_y ** 2)

        qc_b_o = p_b_o_max * aa_b_o * bb_b_o
        qvc_b_o = qc_b_o * r_os_bm_oc_r ** 2 * u_s_oc_norm

        pv_b_o = p_b_o_max * np.sqrt(1 - r_os_bm_oc_x ** 2) * u_s_oc_norm

        qvk_b_o = (c0 * qvc_b_o[:, :, 0:144] +
                   c1 * qvc_b_o[:, :, 144:288] +
                   c2 * qvc_b_o[:, :, 288:432] +
                   c3 * qvc_b_o[:, :, 432:576] +
                   c4 * qvc_b_o[:, :, 576:720])

        qv_b_o = np.zeros((n, 1, 1))
        qv_b_o[:, 0, :] = np.sum(qvk_b_o, axis=2)

        sv_b_o = np.zeros((n, 1, 1))
        sv_b_o[:, 0, :] = np.sum(qvk_b_o*miu_o_oc_norm, axis=2)

        Int0 = qc_b_o * (miu_o_oc - np.array([[[0], [0], [1]]]))
        Int1 = value_1(r_os_bm_oc, Int0)
        Int2 = value_1(r_os_om_oc, -Int0)

        F_b_o_oc = np.zeros((n, 3, 1))
        F_b_o_oc[:, :, 0] = np.sum(
            (c0*Int0[:, :, 0:144] +
             c1*Int0[:, :, 144:288] +
             c2*Int0[:, :, 288:432] +
             c3*Int0[:, :, 432:576] +
             c4*Int0[:, :, 576:720]),
             axis=2
        )

        M_b_o_oc = np.zeros((n, 3, 1))
        M_b_o_oc[:, :, 0] = np.sum(
            (c0*Int1[:,  :,  0:144] +
             c1*Int1[:,  :,  144:288] +
             c2*Int1[:,  :,  288:432] +
             c3*Int1[:,  :,  432:576] +
             c4*Int1[:,  :,  576:720]),
             axis=2
        )

        M_o_b_oc = np.zeros((n, 3, 1))
        M_o_b_oc[:, :, 0] = np.sum(
            (c0*Int2[:, :, 0:144] +
             c1*Int2[:, :, 144:288] +
             c2*Int2[:, :, 288:432] +
             c3*Int2[:, :, 432:576] +
             c4*Int2[:, :, 576:720]),
             axis=2
        )

        F_b_o_a = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_oc_a[i, :, :]
            t1[:, :] = F_b_o_oc[i, :, :]
            t2 = np.dot(t0, t1)
            F_b_o_a[i, :, :] = t2

        F_b_o_I = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_oc_I[i, :, :]
            t1[:, :] = F_b_o_oc[i, :, :]
            t2 = np.dot(t0, t1)
            F_b_o_I[i, :, :] = t2

        F_o_b_I = np.zeros((1, 3, 1))
        F_o_b_I[0, :, :] = np.sum(-F_b_o_I, axis=0)

        M_b_o_b = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_oc_b[i, :, :]
            t1[:, :] = M_b_o_oc[i, :, :]
            t2 = np.dot(t0, t1)
            M_b_o_b[i, :, :] = t2

        M_o_b_o = np.zeros((1, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_oc_o[i, :, :]
            t1[:, :] = M_o_b_oc[i, :, :]
            t2 = np.dot(t0, t1)
            M_o_b_o[0, :, :] += t2
        #######################################################################
        #         Calculate the slice behavior of ball and inner race         #
        #######################################################################
        r_is_bm_ic_x = np.zeros((n, 1, 720))
        r_is_bm_ic_x[:, 0, 0:144] = r_j_0 * (cos_ang * aa_b_i)[:, 0, :]
        r_is_bm_ic_x[:, 0, 144:288] = r_j_1 * (cos_ang * aa_b_i)[:, 0, :]
        r_is_bm_ic_x[:, 0, 288:432] = r_j_2 * (cos_ang * aa_b_i)[:, 0, :]
        r_is_bm_ic_x[:, 0, 432:576] = r_j_3 * (cos_ang * aa_b_i)[:, 0, :]
        r_is_bm_ic_x[:, 0, 576:720] = r_j_4 * (cos_ang * aa_b_i)[:, 0, :]

        r_is_bm_ic_y = np.zeros((n, 1, 720))
        r_is_bm_ic_y[:, 0, 0:144] = r_j_0 * (sin_ang * bb_b_i)[:, 0, :]
        r_is_bm_ic_y[:, 0, 144:288] = r_j_1 * (sin_ang * bb_b_i)[:, 0, :]
        r_is_bm_ic_y[:, 0, 288:432] = r_j_2 * (sin_ang * bb_b_i)[:, 0, :]
        r_is_bm_ic_y[:, 0, 432:576] = r_j_3 * (sin_ang * bb_b_i)[:, 0, :]
        r_is_bm_ic_y[:, 0, 576:720] = r_j_4 * (sin_ang * bb_b_i)[:, 0, :]

        thetak_b_i = np.arcsin(r_is_bm_ic_x / Pr_sur_rad_i)

        r_is_bm_ic_z = np.zeros((n, 1, 720))
        r_is_bm_ic_z[:, :, 0:144] = (
            R_b + Pr_sur_rad_i * (np.cos(thetak_b_i[:, :, 0:144]) - 1)
        )
        r_is_bm_ic_z[:, :, 144:288] = (
            R_b + Pr_sur_rad_i * (np.cos(thetak_b_i[:, :, 144:288]) - 1)
        )
        r_is_bm_ic_z[:, :, 288:432] = (
            R_b + Pr_sur_rad_i * (np.cos(thetak_b_i[:, :, 288:432]) - 1)
        )
        r_is_bm_ic_z[:, :, 432:576] = (
            R_b + Pr_sur_rad_i * (np.cos(thetak_b_i[:, :, 432:576]) - 1)
        )
        r_is_bm_ic_z[:, :, 576:720] = (
            R_b + Pr_sur_rad_i * (np.cos(thetak_b_i[:, :, 576:720]) - 1)
        )

        r_is_bm_ic = np.zeros((n, 3, 720))
        r_is_bm_ic[:, 0, :] = r_is_bm_ic_x[:, 0, :]
        r_is_bm_ic[:, 1, :] = r_is_bm_ic_y[:, 0, :]
        r_is_bm_ic[:, 2, :] = r_is_bm_ic_z[:, 0, :]

        r_is_im_ic = r_is_bm_ic + r_bm_im_ic
        #######################################################################
        #  Calculate the relative sliding speed between ball and inner race   #
        #######################################################################
        u_r_ic = v_i_ic + value_1(omega_i_ic, r_is_im_ic)
        u_b_ic = v_b_ic + value_1(omega_b_ic, r_is_bm_ic)

        u_s_ic = u_r_ic - u_b_ic
        u_s_ic_norm = np.hypot(u_s_ic[:, 0:1, :] ** 2, u_s_ic[:, 1:2, :] ** 2)
        #######################################################################
        #          Calculate the oil film thickness between ball and          #
        #                          inner race center                          #
        #######################################################################
        r_is_bg_ic_ctr = np.zeros((n, 3, 1))
        r_is_bg_ic_ctr[:, 2, 0] = R_b

        r_is_bm_ic_ctr = r_is_bg_ic_ctr + r_bg_bm_ic

        r_is_im_ic_ctr = r_is_bm_ic_ctr + r_bm_im_ic
        r_is_ig_ic_ctr = r_is_im_ic_ctr# - r_ig_im_ic

        u_r_ic_ctr = v_i_ic + value_1(omega_i_ic, r_is_im_ic_ctr)
        u_b_ic_ctr = v_b_ic + value_1(omega_b_ic, r_is_bm_ic_ctr)

        u_s_ic_ctr = np.abs(u_r_ic_ctr - u_b_ic_ctr)

        u_s_ic_ctr_y = np.zeros((n, 1, 1))
        u_s_ic_ctr_y[:, 0, 0] = u_s_ic_ctr[:, 1, 0]

        spin_roll_b_ic = np.zeros((n, 1, 1))
        for i in range(n):
            if omega_i_ic[i, 0, 0] - omega_b_ic[i, 0, 0] != 0:
                spin_roll_b_ic[i, 0, 0] = (
                    (omega_i_ic[i, 2, 0] - omega_b_ic[i, 2, 0]) /
                    (omega_i_ic[i, 0, 0] - omega_b_ic[i, 0, 0])
                )

        ur_b_ic = np.zeros((n, 1, 1))
        ur_b_ic[:, 0, 0] = np.abs(
            (-omega_b_ic[:, 0, 0] * r_is_bm_ic_ctr[:, 2, 0] -
             omega_i_ic[:, 0, 0] * r_is_im_ic_ctr[:, 2, 0] +
             omega_i_ic[:, 2, 0] * r_is_im_ic_ctr[:, 0, 0]) * 0.5
        )

        sli_roll_b_ic = np.zeros((n,1,12))
        sli_roll_b_ic[:,0,:] = u_s_ic[:,1,:]/ur_b_ic[:,0,:]

        oil_prop_b_i = oil_main(oil_type, temp_i, 0)
        (
            vis_lub_b_i, vis_coeff_0_b_i,
            dvis_lub_b_i, ther_cond_lub_b_i
            ) = (
                oil_prop_b_i[3], oil_prop_b_i[7],
                oil_prop_b_i[9], oil_prop_b_i[6]
                )

        common_params = (
            R_yipu_b_i, ep_b_i, vis_lub_b_i, vis_coeff_0_b_i,
            dvis_lub_b_i, ther_cond_lub_b_i, str_parm, 0
        )

        h_iso_b_i, phit_b_i, sip_b_i = (
            np.zeros((n ,1, 1)), np.zeros((n ,1, 1)), np.zeros((n ,1, 1))
        )
        for i in range(n):
            (
                p_b_i_max_ij, aa_b_i_ij, bb_b_i_ij,
                ur_b_ic_ij, u_s_ic_ctr_y_ij
                ) = (
                    p_b_i_max[i, 0, 0], aa_b_i[i, 0, 0], bb_b_i[i, 0, 0],
                    ur_b_ic[i, 0, 0], u_s_ic_ctr_y[i, 0, 0]
                    )
            h_iso_b_i_ij, phit_b_i_ij, sip_b_i_ij = film_thickness(
                p_b_i_max_ij, aa_b_i_ij, bb_b_i_ij,
                ur_b_ic_ij, u_s_ic_ctr_y_ij,
                *common_params
            )
            h_iso_b_i[i, 0, 0], phit_b_i[i, 0, 0], sip_b_i[i, 0, 0] = (
                h_iso_b_i_ij, phit_b_i_ij, sip_b_i_ij
            )

        h_ts_b_i = h_iso_b_i * phit_b_i

        r_is_bg_b_ctr = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_ic_b[i, :, :]
            t1[:, :] = r_is_bg_ic_ctr[i, :, :]
            t2 = np.dot(t0, t1)
            r_is_bg_b_ctr[i, :, :] = t2

        r_is_ig_i_ctr = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_ic_i[i, :, :]
            t1[:, :] = r_is_ig_ic_ctr[i, :, :]
            t2 = np.dot(t0, t1)
            r_is_ig_i_ctr[i, :, :] = t2
        #######################################################################
        #                   Simple A, B, C, D traction model                  #
        #######################################################################
        if k_b_r_trac_type == 0:
            u_s_ic_norm_abs = np.zeros((n, 1, 720))
            u_s_ic_norm_abs[:, 0, :] = np.abs(u_s_ic_norm[:, 0, :])

            exp_term = np.exp(-C_0 * u_s_ic_norm_abs)

            miu_i_ic_norm = np.zeros((n, 1, 720))
            miu_i_ic_norm = (A_0 + B_0 * u_s_ic_norm_abs) * exp_term + D_0

            miu_i_ic = np.zeros((n, 3, 720))
            for i in range(n):
                for j in range(12):
                    if u_s_ic_norm[i, 0, j] != 0:
                        miu_i_ic[i, 0:2, j] = (
                            u_s_ic[i, 0:2, j] / u_s_ic_norm[i, 0, j] *
                            miu_i_ic_norm[i, 0, j]
                        )
        #######################################################################
        #                     User-defined traction model                     #
        #######################################################################
        elif k_b_r_trac_type == -1:
            pass
            """
            ktc = np.zeros_like(h_ts_b_i)
            ktc[h_ts_b_i <= b_i_limt_film] = 1

            miu_i_ic_norm = sub_traction_coefficient(
                ktc, u_s_ic_norm, ur_b_ic, p_b_i_max,
                aa_b_i, bb_b_i, h_ts_b_i, temp_i, 0
            )

            miu_i_ic = np.zeros((n, 3, 720))
            for i in range(n):
                for j in range(720):
                    if u_s_ic_norm[i, 0, j] != 0:
                        miu_i_ic[i, 0:2, j] = (
                            u_s_ic[i, 0:2, j] / u_s_ic_norm[i, 0, j] *
                            miu_i_ic_norm[i, 0, j]
                        )
            """
        #######################################################################
        #                          Force and moment                           #
        #######################################################################
        r_is_bm_ic_r = np.sqrt(1 - r_is_bm_ic_x ** 2 - r_is_bm_ic_y ** 2)

        qc_b_i = p_b_i_max * aa_b_i * bb_b_i
        qvc_b_i = qc_b_i * r_is_bm_ic_r ** 2 * u_s_ic_norm

        pv_b_i = p_b_i_max * r_is_bm_ic_r * u_s_ic_norm

        qvk_b_i = (c0 * qvc_b_i[:, :, 0:144] +
                   c1 * qvc_b_i[:, :, 144:288] +
                   c2 * qvc_b_i[:, :, 288:432] +
                   c3 * qvc_b_i[:, :, 432:576] +
                   c4 * qvc_b_i[:, :, 576:720])

        qv_b_i = np.zeros((n, 1, 1))
        qv_b_i[:, 0, :] = np.sum(qvk_b_i, axis=2)

        sv_b_i = np.zeros((n, 1, 1))
        sv_b_i[:, 0, :] = np.sum(qvk_b_i*miu_i_ic_norm, axis=2)

        Int3 = qc_b_i * (miu_i_ic - np.array([[[0], [0], [1]]]))
        Int4 = value_1(r_is_bm_ic, Int3)
        Int5 = value_1(r_is_im_ic, -Int3)

        F_b_i_ic = np.zeros((n, 3, 1))
        F_b_i_ic[:, :, 0] = np.sum(
            (c0*Int3[:, :, 0:144] +
             c1*Int3[:, :, 144:288] +
             c2*Int3[:, :, 288:432] +
             c3*Int3[:, :, 432:576] +
             c4*Int3[:, :, 576:720]),
             axis=2
        )

        M_b_i_ic = np.zeros((n, 3, 1))
        M_b_i_ic[:, :, 0] = np.sum(
            (c0*Int4[:, :, 0:144] +
             c1*Int4[:, :, 144:288] +
             c2*Int4[:, :, 288:432] +
             c3*Int4[:, :, 432:576] +
             c4*Int4[:, :, 576:720]),
             axis=2
        )

        M_i_b_ic = np.zeros((n, 3, 1))
        M_i_b_ic[:, :, 0] = np.sum(
            (c0*Int5[:, :, 0:144] +
             c1*Int5[:, :, 144:288] +
             c2*Int5[:, :, 288:432] +
             c3*Int5[:, :, 432:576] +
             c4*Int5[:, :, 576:720]),
             axis=2
        )

        F_b_i_a = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_ic_a[i, :, :]
            t1[:, :] = F_b_i_ic[i, :, :]
            t2 = np.dot(t0, t1)
            F_b_i_a[i, :, :] = t2

        F_b_i_I = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_ic_I[i, :, :]
            t1[:, :] = F_b_i_ic[i, :, :]
            t2 = np.dot(t0, t1)
            F_b_i_I[i, :, :] = t2

        F_i_b_I = np.zeros((1, 3, 1))
        F_i_b_I[0, :, :] = np.sum(-F_b_i_I, axis=0)

        M_b_i_b = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_ic_b[i, :, :]
            t1[:, :] = M_b_i_ic[i, :, :]
            t2 = np.dot(t0, t1)
            M_b_i_b[i, :, :] = t2

        M_i_b_i = np.zeros((1, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_ic_i[i, :, :]
            t1[:, :] = M_i_b_ic[i, :, :]
            t2 = np.dot(t0, t1)
            M_i_b_i[0, :, :] += t2
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    Info_brtf = (v_o_I,
                 # Outer race velocity.
                 v_i_I,
                 # Inner race velocity.
                 omega_b_b,
                 # Ball angular velocity.
                 omega_o_o,
                 # Outer race angular velocity.
                 omega_i_i,
                 # Inner race angular velocity.
                 zero_point_o_l,
                 # X/a coordinates for left points of outer race pure rolling.
                 zero_point_o_r,
                 # X/a coordinates for right points of outer race pure rolling.
                 zero_point_i_l,
                 # X/a coordinates for left points of inner race pure rolling.
                 zero_point_i_r,
                 # X/a coordinates for right points of inner race pure rolling.
                 u_s_oc,
                 # Relative slip velocity between ball and
                 # outer race in outer race contact frame.
                 u_s_ic,
                 # Relative slip velocity between ball and outer race in
                 # inner race contact frame.
                 spin_roll_b_oc,
                 # Spin-roll-ratio between ball and outer race in
                 # outer race contact frame.
                 spin_roll_b_ic,
                 # Spin-roll-ratio between ball and inner race in
                 # inner race contact frame.
                 sli_roll_b_oc,
                 # Slide-roll-ratio between ball and outer race in
                 # outer race contact frame.
                 sli_roll_b_ic,
                 # Slide-roll-ratio between ball and inner race in
                 # inner race contact frame.
                 h_ts_b_o,
                 # Film thickness between ball and outer race.
                 h_ts_b_i,
                 # Film thickness between ball and inner race.
                 r_os_bg_b_ctr,
                 # Postion of outer race contact center relative to ball
                 # geometry center in ball frame.
                 r_is_bg_b_ctr,
                 # Postion of inner race contact center relative to ball
                 # geometry center in ball frame.
                 r_os_og_o_ctr,
                 # Postion of outer race contact center relative to outer race
                 # geometry center in outer race frame.
                 r_is_ig_i_ctr,
                 # Postion of inner race contact center relative to inner race
                 # geometry center in inner race frame.
                 miu_o_oc,
                 # Traction coefficient between ball and outer race in
                 # outer race contact frame.
                 miu_i_ic,
                 # Traction coefficient between ball and outer race in
                 # inner race contact frame.
                 pv_b_o,
                 # Stress*velocity value between ball and outer race.
                 pv_b_i,
                 # Stress*velocity value between ball and inner race.
                 qv_b_o,
                 # Force*velocity value between ball and outer race.
                 qv_b_i,
                 # Force*velocity value between ball and inner race.
                 sv_b_o,
                 # Traction*slip value between ball and outer race.
                 sv_b_i
                 # Traction*slip value between ball and inner race.
                 )

    F_b_r = (F_b_o_a,
             # Net force of ball from outer race in azimuth frame.
             F_b_i_a,
             # Net force of ball from inner race in azimuth frame.
             F_o_b_I,
             # Net force of outer race from ball in interial frame.
             F_i_b_I
             # Net force of inner race from ball in interial frame.
             )

    M_b_r = (M_b_o_b,
             # Net moment of ball from outer race in ball frame.
             M_b_i_b,
             # Net moment of ball from inner race in ball frame.
             M_o_b_o,
             # Net moment of outer race from ball in outer race frame.
             M_i_b_i
             # Net moment of inner race from ball in inner race frame.
             )

    return F_b_r, M_b_r, Info_brtf

###############################################################################
#           Calculate the traction force of ball and secondary race           #
###############################################################################
# @njit(fastmath=False)
def ball_race_traction_force_(x, Info_tc, Info_brcs, Info_brcs_, Info_brcf_,
                              Info_brtf, mod_brtf_):
    """Solve the ball and cage force.

    Parameters
    ----------
    x: np.darray
        Solution vector.
    Info_brcs: tuple
        Information of ball_race_contact_strain.
    Info_brcs_: tuple
        Information of ball_race_contact_strain_.
    Info_brcf_: tuple
        Information data of ball_race_contact_force_.
    mod_brtf_: tuple
        Mode data of ball_race_traction_force_.

    Returns
    -------
    F_b_r_: tuple
        Resultant force between the ball and secondary race.
    M_b_r_: tuple
        Resultant moment between the ball and secondary race.
    Info_brtf_: tuple
        Information of ball_race_traction_force_.
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    (A_0,
     B_0,
     C_0,
     D_0,
     D_b,
     R_b,
     R_yipu_b_i,
     R_yipu_b_o,
     Shim_thknss_i,
     Shim_thknss_o,
     b_i_limt_film,
     b_o_limt_film,
     b_r_lub_type,
     dmpg_b_i,
     dmpg_b_o,
     ep_b_i,
     ep_b_o,
     f_i,
     f_o,
     hj,
     k_b_r_trac_type,
     m_b,
     m_i,
     m_o,
     n,
     oil_type,
     r_bg_bm_b,
     str_parm,
     tj
     ) = mod_brtf_[0::]
    ###########################################################################
    #                               End prepare                               #
    ###########################################################################
    temp_o, temp_i = Info_tc[0:2]
    num_con_b_o_, num_con_b_i_ = Info_brcs_[-2::]

    T_I_a, T_b_I, r_bm_I = Info_brcs[5], Info_brcs[6], Info_brcs[14]
    omega_b_b = Info_brtf[2]

    v_b_a = np.zeros((n, 3, 1))
    v_b_a[:, 0, 0] = x[25:24+12*n:12]
    v_b_a[:, 2, 0] = x[27:24+12*n:12]
    ###########################################################################
    #                        No contact on outer race                         #
    ###########################################################################
    if Shim_thknss_o <= 0 or num_con_b_o_ <= 0:
        (
            zero_point_o_l_,
            zero_point_o_r_,
            u_s_oc_,
            spin_roll_b_oc_,
            sli_roll_b_oc_,
            h_ts_b_o_,
            r_os_bg_b_ctr_,
            r_os_og_o_ctr_,
            miu_o_oc_,
            pv_b_o_,
            qv_b_o_,
            sv_b_o_
            ) = (
                np.zeros((n, 1, 1)),
                np.zeros((n, 1, 1)),
                np.zeros((n, 3, 12)),
                np.zeros((n, 1, 1)),
                np.zeros((n, 1, 12)),
                np.zeros((n, 1, 1)),
                np.zeros((n, 3, 1)),
                np.zeros((n, 3, 1)),
                np.zeros((n, 3, 12)),
                np.zeros((n, 1, 12)),
                np.zeros((n, 1, 1)),
                np.zeros((n, 1, 1)),
                )

        (
            F_b_o_a_, F_o_b_I_,
            M_b_o_b_, M_o_b_o_
            ) = (
                np.zeros((n, 3, 1)), np.zeros((1, 3, 1)),
                np.zeros((n, 3, 1)), np.zeros((1, 3, 1))
                )
    ###########################################################################
    #               Secondary outer race for shim thickness > 0               #
    ###########################################################################
    else:
        T_o_I, T_I_o, r_om_I = Info_brcs[0], Info_brcs[1], Info_brcs[10]
        T_a_oc_ = Info_brcs_[0]
        (delta_b_o_, aa_b_o_, bb_b_o_,
         p_b_o_max_, Q_b_o_, Q_b_o_) = Info_brcf_[0:12:2]
        v_o_I, omega_o_o = Info_brtf[0:6:3]

        Pr_sur_rad_o = 2 * f_o * D_b / (2 * f_o + 1)

        T_oc_a_ = np.transpose(T_a_oc_, (0,2,1))

        T_I_oc_ = np.zeros((n, 3, 3))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 3))

        for i in range(n):
            t0[:, :] = T_a_oc_[i, :, :]
            t1[:, :] = T_I_a[i, :, :]
            t2 = np.dot(t0, t1)
            T_I_oc_[i, :, :] = t2

        T_oc_I_ = np.transpose(T_I_oc_, (0,2,1))

        T_oc_o_ = np.zeros((n, 3, 3))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 3))

        for i in range(n):
            t0[:, :] = T_I_o[0, :, :]
            t1[:, :] = T_oc_I_[i, :, :]
            t2 = np.dot(t0, t1)
            T_oc_o_[i, :, :] = t2

        T_b_oc_ = np.zeros((n, 3, 3))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 3))

        for i in range(n):
            t0[:, :] = T_I_oc_[i, :, :]
            t1[:, :] = T_b_I[i, :, :]
            t2 = np.dot(t0, t1)
            T_b_oc_[i, :, :] = t2

        T_oc_b_ = np.transpose(T_b_oc_, (0,2,1))

        u_r_oc_0_ = np.zeros((n, 3, 1))
        u_r_oc_0_[:, 0, 0] = x[29:24+12*n:12]

        v_b_oc_ = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_a_oc_[i, :, :]
            t1[:, :] = v_b_a[i, :, :]
            t2 = np.dot(t0, t1)
            v_b_oc_[i, :, :] = t2

        omega_b_oc_ = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_b_oc_[i, :, :]
            t1[:, :] = omega_b_b[i, :, :]
            t2 = np.dot(t0, t1)
            omega_b_oc_[i, :, :] = t2

        v_o_oc_ = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_I_oc_[i, :, :]
            t1[:, :] = v_o_I[0, :, :]
            t2 = np.dot(t0, t1)
            v_o_oc_[i, :, :] = t2

        omega_o_oc_ = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))
        t2 = np.zeros((3, 1))
        t3 = np.zeros((3, 3))

        for i in range(n):
            t0[:, :] = T_o_I[0, :, :]
            t1[:, :] = omega_o_o[0, :, :]
            t2[:, :] = u_r_oc_0_[i, :, :]
            t3[:, :] = T_I_oc_[i, :, :]
            t4 = np.dot(t0, t1) - t2
            t5 = np.dot(t3, t4)
            omega_o_oc_[i, :, :] = t5

        r_bm_om_oc_ = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_I_oc_[i, :, :]
            t1[:, :] = (r_bm_I - r_om_I)[i, :, :]
            t2 = np.dot(t0, t1)
            r_bm_om_oc_[i, :, :] = t2

        r_bg_bm_oc_ = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_b_oc_[i, :, :]
            t1[:, :] = r_bg_bm_b[i, :, :]
            t2 = np.dot(t0, t1)
            r_bg_bm_oc_[i, :, :] = t2
        #######################################################################
        #    Calculate the slice behavior of ball and secondary outer race    #
        #######################################################################
        zero_point_o_l_, zero_point_o_r_ = (
            np.zeros((n, 1, 1)), np.zeros((n, 1, 1))
        )
        for j in range(n):
            if aa_b_o_[j, 0, 0] > 0:
                common_params = (
                    Pr_sur_rad_o, R_b, aa_b_o_[j, 0, 0],
                    v_o_oc_[j, 1, 0], v_b_oc_[j, 1, 0],
                    omega_o_oc_[j, 0, 0], omega_o_oc_[j, 2, 0],
                    r_bm_om_oc_[j, 0, 0], r_bm_om_oc_[j, 2, 0],
                    omega_b_oc_[j, 0, 0], omega_b_oc_[j, 2, 0],
                    r_bg_bm_oc_[j, 0, 0], r_bg_bm_oc_[j, 2, 0]
                )
                zero_point__1_o_ = slip_zero(-1., *common_params)
                zero_point_0_o_ = slip_zero(0., *common_params)
                zero_point_1_o_ = slip_zero(1., *common_params)
                if zero_point__1_o_ * zero_point_0_o_ < 0:
                    zero_point_o_l_[j, 0, 0] = solve_slip_zero(
                        -1., 0., *common_params
                    )
                else:
                    zero_point_o_l_[j, 0, 0] = 0.
                if zero_point_0_o_ * zero_point_1_o_ <= 0:
                    zero_point_o_r_[j, 0, 0] = solve_slip_zero(
                        0., 1., *common_params
                    )
                else:
                    zero_point_o_r_[j, 0, 0] = 0.

        kkk0_, kkk1_, kkk2_ = value_2(zero_point_o_l_, zero_point_o_r_, tj)

        kkk_ = np.zeros((n, 1, 12))
        kkk_[:, :, 0:4], kkk_[:, :, 4:8], kkk_[:, :, 8:12] = (
            kkk0_, kkk1_, kkk2_
        )

        hhh_ = np.zeros((n, 1, 12))
        hhh_[:, :, 0:4] = (zero_point_o_l_ + 1) * 0.5 * hj
        hhh_[:, :, 4:8] = (zero_point_o_r_ - zero_point_o_l_) * 0.5 * hj
        hhh_[:, :, 8:12] = (1 - zero_point_o_r_) * 0.5 * hj

        thetak_b_o_ = np.arcsin(kkk_ * aa_b_o_ / Pr_sur_rad_o)

        r_os_bg_oc_x_ = kkk_ * aa_b_o_
        r_os_bg_oc_y_ = 0 * kkk_ * aa_b_o_
        r_os_bg_oc_z_ = R_b + Pr_sur_rad_o * (np.cos(thetak_b_o_) - 1)

        r_os_bg_oc_ = np.zeros((n, 3, 12))
        r_os_bg_oc_[:, 0, :] = r_os_bg_oc_x_[:, 0, :]
        r_os_bg_oc_[:, 1, :] = r_os_bg_oc_y_[:, 0, :]
        r_os_bg_oc_[:, 2, :] = r_os_bg_oc_z_[:, 0, :]

        r_os_bm_oc_ = r_os_bg_oc_ + r_bg_bm_oc_
        r_os_om_oc_ = r_os_bm_oc_ + r_bm_om_oc_
        #######################################################################
        #                Calculate the relative sliding speed                 #
        #                between ball and secondary outer race                #
        #######################################################################
        u_r_oc_ = v_o_oc_ + value_1(omega_o_oc_, r_os_om_oc_)
        u_b_oc_ = v_b_oc_ + value_1(omega_b_oc_, r_os_bm_oc_)

        u_s_oc_ = u_r_oc_ - u_b_oc_

        sin_theta = np.sin(thetak_b_o_)
        cos_theta = np.cos(thetak_b_o_)

        u_s_ol_ = np.zeros((n, 3, 12))
        u_s_ol_[:, 0, :] = (u_s_oc_[:, 0, :] * cos_theta[:, 0, :] -
                            u_s_oc_[:, 2, :] * sin_theta[:, 0, :])
        u_s_ol_[:, 1, :] = u_s_oc_[:, 1, :]
        u_s_ol_[:, 2, :] = (u_s_oc_[:, 0, :] * sin_theta[:, 0, :] +
                            u_s_oc_[:, 2, :] * cos_theta[:, 0, :])

        u_s_ol_norm_ = np.hypot(u_s_ol_[:, 0:1, :], u_s_ol_[:, 1:2, :])
        #######################################################################
        #          Calculate the oil film thickness between ball and          #
        #                     secondary outer race center                     #
        #######################################################################
        r_os_bg_oc_ctr_ = np.zeros((n, 3, 1))
        r_os_bg_oc_ctr_[:, 2, 0] = R_b

        r_os_bm_oc_ctr_ = r_os_bg_oc_ctr_ + r_bg_bm_oc_

        r_os_om_oc_ctr_ = r_os_bm_oc_ctr_ + r_bm_om_oc_
        r_os_og_oc_ctr_ = r_os_om_oc_ctr_# - r_og_om_oc_

        u_r_oc_ctr_ = v_o_oc_ + value_1(omega_o_oc_, r_os_om_oc_ctr_)
        u_b_oc_ctr_ = v_b_oc_ + value_1(omega_b_oc_, r_os_bm_oc_ctr_)

        u_s_oc_ctr_ = np.abs(u_r_oc_ctr_ - u_b_oc_ctr_)

        u_s_oc_ctr_y_ = np.zeros((n, 1, 1))
        u_s_oc_ctr_y_[:, 0, :] = u_s_oc_ctr_[:, 1, :]

        spin_roll_b_oc_ = np.zeros((n, 1, 1))
        for i in range(n):
            if omega_o_oc_[i, 0, 0] - omega_b_oc_[i, 0, 0] != 0:
                spin_roll_b_oc_[i, 0, 0] = (
                    (omega_o_oc_[i, 2, 0] - omega_b_oc_[i, 2, 0]) /
                    (omega_o_oc_[i, 0, 0] - omega_b_oc_[i, 0, 0])
                )

        ur_b_oc_ = np.zeros((n, 1, 1))
        ur_b_oc_[:, 0, 0] = np.abs(
            (-omega_b_oc_[:, 0, 0] * r_os_bm_oc_ctr_[:, 2, 0] -
             omega_o_oc_[:, 0, 0] * r_os_om_oc_ctr_[:, 2, 0] +
             omega_o_oc_[:, 2, 0] * r_os_om_oc_ctr_[:, 0, 0]) * 0.5
        )

        sli_roll_b_oc_ = np.zeros((n, 1, 12))
        sli_roll_b_oc_[:, 0, :] = u_s_oc_[:, 1, :] / ur_b_oc_[:, 0, :]

        oil_prop_b_o_ = oil_main(oil_type, temp_o, 0)
        (
            vis_lub_b_o_, vis_coeff_0_b_o_,
            dvis_lub_b_o_, ther_cond_lub_b_o_
            ) = (
                oil_prop_b_o_[3], oil_prop_b_o_[7],
                oil_prop_b_o_[9], oil_prop_b_o_[6]
                )

        common_params = (
            R_yipu_b_o, ep_b_o, vis_lub_b_o_, vis_coeff_0_b_o_,
            dvis_lub_b_o_, ther_cond_lub_b_o_, str_parm, 0
        )

        h_iso_b_o_, phit_b_o_, sip_b_o_ = (
            np.zeros((n ,1, 1)), np.zeros((n ,1, 1)), np.zeros((n ,1, 1))
        )
        for i in range(n):
            (
                p_b_o_max_ij_, aa_b_o_ij_, bb_b_o_ij_,
                ur_b_oc_ij_, u_s_oc_ctr_y_ij_
                ) = (
                    p_b_o_max_[i, 0, 0], aa_b_o_[i, 0, 0], bb_b_o_[i, 0, 0],
                    ur_b_oc_[i, 0, 0], u_s_oc_ctr_y_[i, 0, 0]
                    )
            h_iso_b_o_ij_, phit_b_o_ij_, sip_b_o_ij_ = film_thickness(
                p_b_o_max_ij_, aa_b_o_ij_, bb_b_o_ij_,
                ur_b_oc_ij_, u_s_oc_ctr_y_ij_,
                *common_params
            )
            h_iso_b_o_[i, 0, 0], phit_b_o_[i, 0, 0], sip_b_o_[i, 0, 0] = (
                h_iso_b_o_ij_, phit_b_o_ij_, sip_b_o_ij_
            )

        h_ts_b_o_ = h_iso_b_o_ * phit_b_o_

        r_os_bg_b_ctr_ = np.zeros((n, 3, 1))
        r_os_og_o_ctr_ = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))
        t2 = np.zeros((3, 3))
        t3 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_oc_b_[i, :, :]
            t1[:, :] = r_os_bg_oc_ctr_[i, :, :]
            t2[:, :] = T_oc_o_[i, :, :]
            t3[:, :] = r_os_og_oc_ctr_[i, :, :]
            t4 = np.dot(t0, t1)
            t5 = np.dot(t2, t3)
            r_os_bg_b_ctr_[i, :, :] = t4
            r_os_og_o_ctr_[i, :, :] = t5
        #######################################################################
        #                   Simple A, B, C, D traction model                  #
        #######################################################################
        if k_b_r_trac_type == 0:
            u_s_ol_norm_abs_ = np.zeros((n, 1, 12))
            u_s_ol_norm_abs_[:, 0, :] = np.abs(u_s_ol_norm_[:, 0, :])

            exp_term = np.exp(-C_0 * u_s_ol_norm_abs_)

            miu_o_ol_norm_ = np.zeros((n, 1, 12))
            miu_o_ol_norm_ = (A_0 + B_0 * u_s_ol_norm_abs_) * exp_term + D_0

            miu_o_ol_ = np.zeros((n, 3, 12))
            for i in range(n):
                for j in range(12):
                    if u_s_ol_norm_[i, 0, j] != 0:
                        miu_o_ol_[i, 0:2, j] = (
                            u_s_ol_[i, 0:2, j] / u_s_ol_norm_[i, 0, j] *
                            miu_o_ol_norm_[i, 0, j]
                        )

            miu_o_oc_ = np.zeros((n, 3, 12))
            miu_o_oc_[:, 0:2, :] = miu_o_ol_[:, 0:2, :]
        #######################################################################
        #                     User-defined traction model                     #
        #######################################################################
        elif k_b_r_trac_type == -1:
            pass
            """
            ktc = np.zeros_like(h_ts_b_o_)
            ktc[h_ts_b_o_ <= b_o_limt_film] = 1

            miu_o_ol_norm_ = sub_traction_coefficient(
                ktc, u_s_ol_norm_, ur_b_oc_, p_b_o_max_,
                aa_b_o_, bb_b_o_, h_ts_b_o_, temp_o, 0
            )

            miu_o_ol_ = np.zeros((n, 3, 12))
            for i in range(n):
                for j in range(720):
                    if u_s_ol_norm_[i, 0, j] != 0:
                        miu_o_ol_[i, 0:2, j] = (
                            u_s_ol_[i, 0:2, j] / u_s_ol_norm_[i, 0, j] *
                            miu_o_ol_norm_[i, 0, j]
                        )

            miu_o_oc_ = np.zeros((n, 3, 12))
            miu_o_oc_[:, 0:2, :] = miu_o_ol_[:, 0:2, :]
            """
        #######################################################################
        #                          Force and moment                           #
        #######################################################################
        k_b_o_ = np.zeros((n, 1, 1))
        for i in range(n):
            if delta_b_o_[i, 0, 0] > 0:
                k_b_o_[i, 0, 0] = (
                    1.5 * Q_b_o_[i, 0, 0] / delta_b_o_[i, 0, 0]
                )

        brdq_b_o_ = damping_coefficient(
            dmpg_b_o, 0, m_b, m_o, k_b_o_, Q_b_o_, 1, 0
        )

        df_b_o_ = (
            math.pi * 0.5 * p_b_o_max_ * aa_b_o_ * bb_b_o_ * (1 - kkk_ ** 2)
        )

        pv_b_o_ = p_b_o_max_ * np.sqrt(1 - kkk_ ** 2) * u_s_ol_norm_
        qvk_b_o_ = df_b_o_ * u_s_ol_norm_ * hhh_

        qv_b_o_ = np.zeros((n, 1, 1))
        qv_b_o_[:, :, 0] = np.sum(qvk_b_o_, axis=2)

        sv_b_o_ = np.zeros((n, 1, 1))
        sv_b_o_[:, :, 0] = np.sum(qvk_b_o_*miu_o_ol_norm_, axis=2)

        unit_array_z = np.array([[[0], [0], [1]]])

        Int0_ = (df_b_o_ * (miu_o_oc_ - unit_array_z) +
                 brdq_b_o_ * u_s_oc_[:, 2:3, :] * unit_array_z)
        Int0_1 = hhh_ * value_1(r_os_om_oc_, -Int0_)

        Int0_2 = np.zeros((n, 3, 12))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            for j in range(12):
                t0[:, :] = T_oc_o_[i, :, :]
                t1[:, :] = Int0_1[i, :, j:j+1]
                t2 = np.dot(t0, t1)
                Int0_2[i, :, j:j+1] = t2

        F_b_o_oc_ = np.zeros((n, 3, 1))
        F_b_o_oc_[:, :, 0] = np.sum(Int0_*hhh_, axis=2)

        M_b_o_oc_ = np.zeros((n, 3, 1))
        M_b_o_oc_[:, :, 0] = np.sum(hhh_*value_1(r_os_bm_oc_, Int0_), axis=2)

        F_b_o_a_ = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_oc_a_[i, :, :]
            t1[:, :] = F_b_o_oc_[i, :, :]
            t2 = np.dot(t0, t1)
            F_b_o_a_[i, :, :] = t2

        F_b_o_I_ = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_oc_I_[i, :, :]
            t1[:, :] = F_b_o_oc_[i, :, :]
            t2 = np.dot(t0, t1)
            F_b_o_I_[i, :, :] = t2

        F_o_b_I_ = np.zeros((1, 3, 1))
        F_o_b_I_[0, :, :] = np.sum(-F_b_o_I_, axis=0)

        M_b_o_b_ = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_oc_b_[i, :, :]
            t1[:, :] = M_b_o_oc_[i, :, :]
            t2 = np.dot(t0, t1)
            M_b_o_b_[i, :, :] = t2

        M_o_b_o_ = np.zeros((1, 3, 1))
        M_o_b_o_[0, :, 0] = np.sum((np.sum(Int0_2, axis=2)), axis=0)
    ###########################################################################
    #                        No contact on inner race                         #
    ###########################################################################
    if Shim_thknss_i <= 0 or num_con_b_i_ <= 0:
        (
            zero_point_i_l_,
            zero_point_i_r_,
            u_s_ic_,
            spin_roll_b_ic_,
            sli_roll_b_ic_,
            h_ts_b_i_,
            r_is_bg_b_ctr_,
            r_is_ig_i_ctr_,
            miu_i_ic_,
            pv_b_i_,
            qv_b_i_,
            sv_b_i_
            ) = (
                np.zeros((n, 1, 1)),
                np.zeros((n, 1, 1)),
                np.zeros((n, 3, 12)),
                np.zeros((n, 1, 1)),
                np.zeros((n, 1, 12)),
                np.zeros((n, 1, 1)),
                np.zeros((n, 3, 1)),
                np.zeros((n, 3, 1)),
                np.zeros((n, 3, 12)),
                np.zeros((n, 1, 12)),
                np.zeros((n, 1, 1)),
                np.zeros((n, 1, 1)),
                )

        (
            F_b_i_a_, F_i_b_I_,
            M_b_i_b_, M_i_b_i_
            ) = (
                np.zeros((n, 3, 1)), np.zeros((1, 3, 1)),
                np.zeros((n, 3, 1)), np.zeros((1, 3, 1))
                )
    else:
        T_i_I, T_I_i, r_im_I = Info_brcs[2], Info_brcs[3], Info_brcs[12]
        T_a_ic_ = Info_brcs_[1]
        (delta_b_i_, aa_b_i_, bb_b_i_,
         p_b_i_max_, Q_b_i_, Q_b_i_) = Info_brcf_[1:12:2]
        v_i_I, omega_i_i = Info_brtf[1:7:3]

        Pr_sur_rad_i = 2 * f_i * D_b / (2 * f_i + 1)

        T_ic_a_ = np.transpose(T_a_ic_, (0,2,1))

        T_I_ic_ = np.zeros((n, 3, 3))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 3))

        for i in range(n):
            t0[:, :] = T_a_ic_[i, :, :]
            t1[:, :] = T_I_a[i, :, :]
            t2 = np.dot(t0, t1)
            T_I_ic_[i, :, :] = t2

        T_ic_I_ = np.transpose(T_I_ic_, (0,2,1))

        T_ic_I_ = np.zeros((n, 3, 3))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 3))

        for i in range(n):
            t0[:, :] = T_I_i[0, :, :]
            t1[:, :] = T_ic_I_[i, :, :]
            t2 = np.dot(t0, t1)
            T_ic_I_[i, :, :] = t2

        T_b_ic_ = np.zeros((n, 3, 3))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 3))

        for i in range(n):
            t0[:, :] = T_I_ic_[i, :, :]
            t1[:, :] = T_b_I[i, :, :]
            t2 = np.dot(t0, t1)
            T_b_ic_[i, :, :] = t2

        T_ic_b_ = np.transpose(T_b_ic_, (0,2,1))

        u_r_ic_0_ = np.zeros((n, 3, 1))
        u_r_ic_0_[:, 0, 0] = x[29:24+12*n:12]

        v_b_ic_ = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_a_ic_[i, :, :]
            t1[:, :] = v_b_a[i, :, :]
            t2 = np.dot(t0, t1)
            v_b_ic_[i, :, :] = t2

        omega_b_ic_ = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_b_ic_[i, :, :]
            t1[:, :] = omega_b_b[i, :, :]
            t2 = np.dot(t0, t1)
            omega_b_ic_[i,:,:] = t2

        v_i_ic_ = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_I_ic_[i, :, :]
            t1[:, :] = v_i_I[0, :, :]
            t2 = np.dot(t0, t1)
            v_i_ic_[i, :, :] = t2

        omega_i_ic_ = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))
        t2 = np.zeros((3, 1))
        t3 = np.zeros((3, 3))

        for i in range(n):
            t0[:,:] = T_i_I[0, :, :]
            t1[:,:] = omega_i_i[0, :, :]
            t2[:,:] = u_r_ic_0_[i, :, :]
            t3[:,:] = T_I_ic_[i, :, :]
            t4 = np.dot(t0, t1) - t2
            t5 = np.dot(t3, t4)
            omega_i_ic_[i, :, :] = t5

        r_bm_im_ic_ = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_I_ic_[i, :, :]
            t1[:, :] = (r_bm_I - r_im_I)[i, :, :]
            t2 = np.dot(t0, t1)
            r_bm_im_ic_[i, :, :] = t2

        r_bg_bm_ic_ = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_b_ic_[i, :, :]
            t1[:, :] = r_bg_bm_b[i, :, :]
            t2 = np.dot(t0, t1)
            r_bg_bm_ic_[i, :, :] = t2
        #######################################################################
        #   Calculate the slice behavior of ball and  secondary inner race    #
        #######################################################################
        zero_point_i_l_, zero_point_i_r_ = (
            np.zeros((n, 1, 1)), np.zeros((n, 1, 1))
        )
        for j in range(n):
            if aa_b_i_[j, 0, 0] > 0:
                common_params = (
                    Pr_sur_rad_i, R_b, aa_b_i_[j, 0, 0],
                    v_i_ic_[j, 1, 0], v_b_ic_[j, 1, 0],
                    omega_i_ic_[j, 0, 0], omega_i_ic_[j, 2, 0],
                    r_bm_im_ic_[j, 0, 0], r_bm_im_ic_[j, 2, 0],
                    omega_b_ic_[j, 0, 0], omega_b_ic_[j, 2, 0],
                    r_bg_bm_ic_[j, 0, 0], r_bg_bm_ic_[j, 2, 0]
                )
                zero_point__1_i_ = slip_zero(-1., *common_params)
                zero_point_0_i_ = slip_zero(0., *common_params)
                zero_point_1_i_ = slip_zero(1., *common_params)
                if zero_point__1_i_ * zero_point_0_i_ < 0:
                    zero_point_i_l_[j, 0, 0] = solve_slip_zero(
                        -1., 0., *common_params
                    )
                else:
                    zero_point_i_l_[j, 0, 0] = 0.
                if zero_point_0_i_ * zero_point_1_i_ <= 0:
                    zero_point_i_r_[j, 0, 0] = solve_slip_zero(
                        0., 1., *common_params
                    )
                else:
                    zero_point_i_r_[j, 0, 0] = 0.

        k0_, k1_, k2_ = value_2(zero_point_i_l_, zero_point_i_r_, tj)

        k_ = np.zeros((n, 1, 12))
        k_[:, :, 0:4], k_[:, :, 4:8], k_[:, :, 8:12] = k0_, k1_, k2_

        h_ = np.zeros((n, 1, 12))
        h_[:, :, 0:4] = (zero_point_i_l_ + 1) * 0.5 *hj
        h_[:, :, 4:8] = (zero_point_i_r_ - zero_point_i_l_) * 0.5 * hj
        h_[:, :, 8:12] = (1 - zero_point_i_r_) * 0.5 * hj

        thetak_b_i_ = np.arcsin(k_ * aa_b_i_ / Pr_sur_rad_i)

        r_is_bg_ic_x_ = k_ * aa_b_i_
        r_is_bg_ic_y_ = 0 * k_ * aa_b_i_
        r_is_bg_ic_z_ = R_b + Pr_sur_rad_i * (np.cos(thetak_b_i_) - 1)

        r_is_bg_ic_ = np.zeros((n, 3, 12))
        r_is_bg_ic_[:, 0, :] = r_is_bg_ic_x_[:, 0, :]
        r_is_bg_ic_[:, 1, :] = r_is_bg_ic_y_[:, 0, :]
        r_is_bg_ic_[:, 2, :] = r_is_bg_ic_z_[:, 0, :]

        r_is_bm_ic_ = r_is_bg_ic_ + r_bg_bm_ic_
        r_is_im_ic_ = r_is_bm_ic_ + r_bm_im_ic_
        #######################################################################
        #                Calculate the relative sliding speed                 #
        #                between ball and secondary inner race                #
        #######################################################################
        u_r_ic_ = v_i_ic_ + value_1(omega_i_ic_, r_is_im_ic_)
        u_b_ic_ = v_b_ic_ + value_1(omega_b_ic_, r_is_bm_ic_)

        u_s_ic_ = u_r_ic_ - u_b_ic_

        sin_theta = np.sin(thetak_b_i_)
        cos_theta = np.cos(thetak_b_i_)

        u_s_il_ = np.zeros((n, 3, 12))
        u_s_il_[:, 0, :] = (u_s_ic_[:, 0, :] * cos_theta[:, 0, :] -
                            u_s_ic_[:, 2, :] * sin_theta[:, 0, :])
        u_s_il_[:, 1, :] = u_s_ic_[:, 1, :]
        u_s_il_[:, 2, :] = (u_s_ic_[:, 0, :] * sin_theta[:, 0, :] +
                            u_s_ic_[:, 2, :] * cos_theta[:, 0, :])

        u_s_il_norm_ = np.hypot(u_s_il_[:, 0:1, :], u_s_il_[:, 1:2, :])
        #######################################################################
        #          Calculate the oil film thickness between ball and          #
        #                      secondary inner race center                    #
        #######################################################################
        r_is_bg_ic_ctr_ = np.zeros((n, 3, 1))
        r_is_bg_ic_ctr_[:, 2, 0] = R_b

        r_is_bm_ic_ctr_ = r_is_bg_ic_ctr_ + r_bg_bm_ic_

        r_is_im_ic_ctr_ = r_is_bm_ic_ctr_ + r_bm_im_ic_
        r_is_ig_ic_ctr_ = r_is_im_ic_ctr_# - _r_ig_im_ic

        u_r_ic_ctr_ = v_i_ic_ + value_1(omega_i_ic_, r_is_im_ic_ctr_)
        u_b_ic_ctr_ = v_b_ic_ + value_1(omega_b_ic_, r_is_bm_ic_ctr_)

        u_s_ic_ctr_ = np.abs(u_r_ic_ctr_ - u_b_ic_ctr_)

        u_s_ic_ctr_y_ = np.zeros((n, 1, 1))
        u_s_ic_ctr_y_[:, 0, :] = u_s_ic_ctr_[:, 1, :]

        spin_roll_b_ic_ = np.zeros((n, 1, 1))
        for i in range(n):
            if omega_i_ic_[i, 0, 0] - omega_b_ic_[i, 0, 0] != 0:
                spin_roll_b_ic_[i, 0, 0] = (
                    ((omega_i_ic_[i, 2, 0] - omega_b_ic_[i, 2, 0]) /
                     (omega_i_ic_[i, 0, 0] - omega_b_ic_[i, 0, 0]))
                )

        ur_b_ic_ = np.zeros((n, 1, 1))
        ur_b_ic_[:, 0, 0] = np.abs(
            (-omega_b_ic_[:, 0, 0] * r_is_bm_ic_ctr_[:, 2, 0] -
             omega_i_ic_[:, 0, 0] * r_is_im_ic_ctr_[:, 2, 0] +
             omega_i_ic_[:, 2, 0] * r_is_im_ic_ctr_[:, 0, 0]) * 0.5
        )

        sli_roll_b_ic_ = np.zeros((n, 1, 12))
        sli_roll_b_ic_[:, 0, :] = u_s_ic_[:, 1, :] / ur_b_ic_[:, 0, :]

        oil_prop_b_i_ = oil_main(oil_type, temp_i, 0)
        (
            vis_lub_b_i_, vis_coeff_0_b_i_,
            dvis_lub_b_i_, ther_cond_lub_b_i_
            ) = (
                oil_prop_b_i_[3], oil_prop_b_i_[7],
                oil_prop_b_i_[9], oil_prop_b_i_[6]
                )

        common_params = (
            R_yipu_b_i, ep_b_i, vis_lub_b_i_, vis_coeff_0_b_i_,
            dvis_lub_b_i_, ther_cond_lub_b_i_, str_parm, 0
        )

        h_iso_b_i_, phit_b_i_, sip_b_i_ = (
            np.zeros((n ,1, 1)), np.zeros((n ,1, 1)), np.zeros((n ,1, 1))
        )
        for i in range(n):
            (
                p_b_i_max_ij_, aa_b_i_ij_, bb_b_i_ij_,
                ur_b_ic_ij_, u_s_ic_ctr_y_ij_
                ) =(
                    p_b_i_max_[i, 0, 0], aa_b_i_[i, 0, 0], bb_b_i_[i, 0, 0],
                    ur_b_ic_[i, 0, 0], u_s_ic_ctr_y_[i, 0, 0]
                    )
            h_iso_b_i_ij_, phit_b_i_ij_, sip_b_i_ij_ = film_thickness(
                p_b_i_max_ij_, aa_b_i_ij_, bb_b_i_ij_,
                ur_b_ic_ij_, u_s_ic_ctr_y_ij_,
                *common_params
            )
            h_iso_b_i_[i, 0, 0], phit_b_i_[i, 0, 0], sip_b_i_[i, 0, 0] = (
                h_iso_b_i_ij_, phit_b_i_ij_, sip_b_i_ij_
            )

        h_ts_b_i_ = h_iso_b_i_ * phit_b_i_

        r_is_bg_b_ctr_ = np.zeros((n, 3, 1))
        r_is_ig_i_ctr_ = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))
        t2 = np.zeros((3, 3))
        t3 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] =T_ic_b_[i, :, :]
            t1[:, :] = r_is_bg_ic_ctr_[i, :, :]
            t2[:, :] = T_ic_I_[i, :, :]
            t3[:, :] = r_is_ig_ic_ctr_[i, :, :]
            t4 = np.dot(t0, t1)
            t5 = np.dot(t2, t3)
            r_is_bg_b_ctr_[i, :, :] = t4
            r_is_ig_i_ctr_[i, :, :] = t5
        #######################################################################
        #                   Simple A, B, C, D traction model                  #
        #######################################################################
        if k_b_r_trac_type == 0:
            u_s_il_norm_abs_ = np.zeros((n, 1, 12))
            u_s_il_norm_abs_[:, 0, :] = np.abs(u_s_il_norm_[:, 0, :])

            exp_term = np.exp(-C_0 * u_s_il_norm_abs_)

            miu_i_il_norm_ = np.zeros((n, 1, 12))
            miu_i_il_norm_ = (A_0 + B_0 * u_s_il_norm_abs_) * exp_term + D_0

            miu_i_il_ = np.zeros((n, 3, 12))
            for i in range(n):
                for j in range(12):
                    if u_s_il_norm_[i, 0, j] != 0:
                        miu_i_il_[i, 0:2, j] = (
                            u_s_il_[i, 0:2, j] / u_s_il_norm_[i, 0, j] *
                            miu_i_il_norm_[i, 0, j]
                        )

            miu_i_ic_ = np.zeros((n, 3, 12))
            miu_i_ic_[:, 0:2, :] = miu_i_il_[:, 0:2, :]
        #######################################################################
        #                     User-defined traction model                     #
        #######################################################################
        elif k_b_r_trac_type == -1:
            pass
            """
            ktc = np.zeros_like(h_ts_b_i_)
            ktc[h_ts_b_i_ <= b_i_limt_film] = 1

            miu_i_il_norm_ = sub_traction_coefficient(
                ktc, u_s_il_norm_, ur_b_ic_, p_b_i_max_,
                aa_b_i_, bb_b_i_, h_ts_b_i_, temp_i, 0
            )

            miu_i_il_ = np.zeros((n, 3, 12))
            for i in range(n):
                for j in range(720):
                    if u_s_il_norm_[i, 0, j] != 0:
                        miu_i_il_[i, 0:2, j] = (
                            u_s_il_[i, 0:2, j] / u_s_il_norm_[i, 0, j] *
                            miu_i_il_norm_[i, 0, j]
                        )

            miu_i_ic_ = np.zeros((n, 3, 12))
            miu_i_ic_[:, 0:2, :] = miu_i_il_[:, 0:2, :]
            """
        #######################################################################
        #                          Force and moment                           #
        #######################################################################
        k_b_i_ = np.zeros((n, 1, 1))
        for i in range(n):
            if delta_b_i_[i, 0, 0] > 0:
                k_b_i_[i, 0, 0] = 1.5 * Q_b_i_[i, 0, 0] / delta_b_i_[i, 0, 0]

        brdq_b_i_ = damping_coefficient(
            dmpg_b_i, 0, m_b, m_i, k_b_i_, Q_b_i_, 1, 0
        )

        df_b_i_ = (
            math.pi * 0.5 * p_b_i_max_ * aa_b_i_ * bb_b_i_ * (1 - k_ ** 2)
        )

        pv_b_i_ = p_b_i_max_ * np.sqrt(1 - k_ ** 2) * u_s_il_norm_
        qvk_b_i_ = df_b_i_ * u_s_il_norm_ * h_

        qv_b_i_ = np.zeros((n, 1, 1))
        qv_b_i_[:, :, 0] = np.sum(qvk_b_i_, axis=2)

        sv_b_i_ = np.zeros((n, 1, 1))
        sv_b_i_[:, :, 0] = np.sum(qvk_b_i_*miu_i_il_norm_, axis=2)

        unit_array_z = np.array([[[0], [0], [1]]])

        Int1_ = (df_b_i_ * (miu_i_ic_ - unit_array_z) +
                 brdq_b_i_ * u_s_ic_[:, 2:3, :] * unit_array_z)
        Int1_1 = h_ * value_1(r_is_im_ic_, -Int1_)

        Int1_2 = np.zeros((n, 3, 12))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            for j in range(12):
                t0[:, :] = T_ic_I_[i, :, :]
                t1[:, :] = Int1_1[i, :, j:j+1]
                t2 = np.dot(t0, t1)
                Int1_2[i, :, j:j+1] = t2

        F_b_i_ic_ = np.zeros((n, 3, 1))
        F_b_i_ic_[:, :, 0] = np.sum(Int1_*h_, axis=2)

        M_b_i_ic_ = np.zeros((n, 3, 1))
        M_b_i_ic_[:, :, 0] = np.sum(h_*value_1(r_is_bm_ic_, Int1_), axis=2)

        F_b_i_a_ = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_ic_a_[i, :, :]
            t1[:, :] = F_b_i_ic_[i, :, :]
            t2 = np.dot(t0, t1)
            F_b_i_a_[i, :, :] = t2

        F_b_i_I_ = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_ic_I_[i, :, :]
            t1[:, :] = F_b_i_ic_[i, :, :]
            t2 = np.dot(t0, t1)
            F_b_i_I_[i, :, :] = t2

        F_i_b_I_ = np.zeros((1, 3, 1))
        F_i_b_I_[0, :, :] = np.sum(-F_b_i_I_, axis=0)

        M_b_i_b_  = np.zeros((n, 3, 1))
        t0 = np.zeros((3, 3))
        t1 = np.zeros((3, 1))

        for i in range(n):
            t0[:, :] = T_ic_b_[i, :, :]
            t1[:, :] = M_b_i_ic_[i, :, :]
            t2 = np.dot(t0, t1)
            M_b_i_b_[i, :, :] = t2

        M_i_b_i_ = np.zeros((1, 3, 1))
        M_i_b_i_[0, :, 0] = np.sum((np.sum(Int1_2, axis=2)), axis=0)
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    Info_brtf_ = (zero_point_o_l_,
                  # X/a coordinates for left points of secondary outer race
                  # pure rolling.
                  zero_point_o_r_,
                  # X/a coordinates for right points of secondary outer race
                  # pure rolling.
                  zero_point_i_l_,
                  # X/a coordinates for left points of secondary inner race
                  # pure rolling.
                  zero_point_i_r_,
                  # X/a coordinates for right points of secondary inner race
                  # pure rolling.
                  u_s_oc_,
                  # Relative slip velocity between ball and secondary outer
                  # race in secondary outer race contact frame.
                  u_s_ic_,
                  # Relative slip velocity between ball and secondary inner
                  # race in secondary inner race contact frame.
                  spin_roll_b_oc_,
                  # Spin-roll-ratio between ball and secondary outer race in
                  # secondary outer race contact frame.
                  spin_roll_b_ic_,
                  # Spin-roll-ratio between ball and secondary inner race in
                  # secondary inner race contact frame.
                  sli_roll_b_oc_,
                  # Slide-roll-ratio between ball and outer race in
                  # secondary outer race contact frame.
                  sli_roll_b_ic_,
                  # Slide-roll-ratio between ball and inner race in
                  # secondary inner race contact frame.
                  h_ts_b_o_,
                  # Film thickness between ball and secondary outer race in
                  # secondary outer race contact frame.
                  h_ts_b_i_,
                  # Film thickness between ball and secondary inner race in
                  # secondary inner race contact frame.
                  r_os_bg_b_ctr_,
                  # Postion of secondary outer race contact center relative to
                  # ball geometry center in ball frame.
                  r_is_bg_b_ctr_,
                  # Postion of secondary inner race contact center relative to
                  # secondary inner race geometry center in secondary inner 
                  # ace frame.
                  r_os_og_o_ctr_,
                  # Postion of secondary outer race contact center relative to
                  # secondary outer race geometry center in secondary outer
                  # race frame.
                  r_is_ig_i_ctr_,
                  # Postion of secondary inner race contact center relative to
                  # secondary inner race geometry center in secondary inner
                  # race frame.
                  miu_o_oc_,
                  # Traction coefficient between ball and secondary outer race
                  # in secondary outer race contact frame.
                  miu_i_ic_,
                  # Traction coefficient between ball and secondary inner race
                  # in secondary inner race contact frame.
                  pv_b_o_,
                  # Stress*velocity value between ball and secondary outer
                  # race.
                  pv_b_i_,
                  # Stress*velocity value between ball and secondary inner
                  # race.
                  qv_b_o_,
                  # Force*velocity value between ball and secondary outer race.
                  qv_b_i_,
                  # Force*velocity value between ball and secondary inner race.
                  sv_b_o_,
                  # Traction*slip value between ball and secondary outer race.
                  sv_b_i_
                  # Traction*slip value between ball and secondary inner race.
                  )

    F_b_r_ = (F_b_o_a_,
              # Net force of ball from secondary outer race in azimuth frame.
              F_b_i_a_,
              # Net force of ball from secondary inner race in azimuth frame.
              F_o_b_I_,
              # Net force of secondary outer race from ball in interial frame.
              F_i_b_I_
              # Net force of secondary inner race from ball in interial frame.
              )

    M_b_r_ = (M_b_o_b_,
              # Net moment of ball from secondary outer race in ball frame.
              M_b_i_b_,
              # Net moment of ball from secondary inner race in ball frame.
              M_o_b_o_,
              # Net moment of secondary outer race from ball in secondary
              # outer race frame.
              M_i_b_i_
              # Net moment of secondary inner race from ball in secondary
              # inner race frame.
              )

    return F_b_r_, M_b_r_, Info_brtf_

###############################################################################
#              Calculate load and moment between different ball               #
###############################################################################
def ball_ball_force(x, Info_brcs, Info_brtf, mod_bbf):
    """Solve load and moment between different ball.

    Parameters
    ----------
    x: np.darray
        Solution vector.
    Info_brcs: tuple
        Information of ball_race_contact_strain.
    Info_brtf: tuple
        Information of traction_force.
    mod_bbf: tuple
        Mode data of ball_ball_force.

    Returns
    -------
    F_b_b: tuple
        Resultant force between the jth ball and (j+1)th ball.
    M_b_b: tuple
        Resultant moment between the jth ball and (j+1)th ball.
    Info_bbf: tuple
        Information of ball_ball_force.
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    (D_b,
     E_b,
     R_b,
     b_b_trac_type,
     b_b_limt_film,
     elas_stra_limt_b,
     f_b_b,
     n,
     po_b
     ) = mod_bbf[0::]
    ###########################################################################
    #                               End prepare                               #
    ###########################################################################
    T_I_a, T_b_I, T_I_b, r_bg_I = (
        Info_brcs[5], Info_brcs[6], Info_brcs[7], Info_brcs[15]
    )
    omega_b_b = Info_brtf[2]

    omega_b_a = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 3))
    t2 = np.zeros((3, 1))

    for i in range(n):
        t0[:, :] = T_I_a[i, :, :]
        t1[:, :] = T_b_I[i, :, :]
        t2[:, :] = omega_b_b[i, :, :]
        t3 = np.dot(t0, np.dot(t1, t2))
        omega_b_a[i, :, :] = t3

    deltah = b_b_limt_film
    ###########################################################################
    #                       Strain and relative velocity                      #
    ###########################################################################
    r_bg_bg_I = np.zeros((n, 3, 1))
    r_bg_bg_I[0:-1, :, :] = np.diff(r_bg_I, n=1, axis=1)
    r_bg_bg_I[-1, :, :] = r_bg_I[0, :, :] - r_bg_I[-1, :, :]

    r_bg_bg_norm = np.zeros((n, 1, 1))
    r_bg_bg_norm[:, 0, 0] = (r_bg_bg_I[:, 0, 0] ** 2 +
                             r_bg_bg_I[:, 1, 0] ** 2 +
                             r_bg_bg_I[:, 2, 0] ** 2) ** 0.5

    e_bg_bg_norm = r_bg_bg_I / r_bg_bg_norm

    delta_b_b = r_bg_bg_norm - D_b

    r0 = R_b * e_bg_bg_norm
    u0, u1 = value_1(omega_b_a, r0), value_1(omega_b_a, -r0)

    u_b_b_a = u1 - u0
    u_b_b_a_norm = np.zeros((n, 1, 1))
    u_b_b_a_norm[:, 0, 0] = (u_b_b_a[:, 0, 0] ** 2 +
                             u_b_b_a[:, 1, 0] ** 2 +
                             u_b_b_a[:, 2, 0] ** 2) ** 0.5

    e_u_b_b_a_norm = u_b_b_a / u_b_b_a_norm
    ###########################################################################
    #                            Plastic contact                              #
    ###########################################################################
    s_b_b = elas_stra_limt_b * E_b

    delta_b_b = np.zeros((n, 1, 1))
    for i in range(n):
        if delta_b_b[i, 0, 0] < deltah:
            delta_b_b[i, 0, 0] = np.abs(delta_b_b[i, 0, 0] - deltah)

    arc = 0.5 * math.pi * R_b * delta_b_b
    Q_b_b = E_b * D_b ** 0.5 / 3 / (1 - po_b ** 2) * delta_b_b ** 1.5
    k_b_b = 1.5 * Q_b_b / delta_b_b
    st = Q_b_b / arc
    for i in range(n):
        if st[i, 0, 0] > s_b_b:
            Q_b_b[i, 0, 0] = arc * s_b_b[i, 0, 0]
            k_b_b[i, 0, 0] = Q_b_b[i, 0, 0] / delta_b_b[i, 0, 0]
    ###########################################################################
    #                          Simple traction model                          #
    ###########################################################################
    if b_b_trac_type == 0:
        miu_b_b_norm = f_b_b
    ###########################################################################
    #                       User-defined traction model                       #
    ###########################################################################
    elif b_b_trac_type == -1:
        pass
        """
        ur_b_b_a_norm = 0.5 * (u2 + u1)
        ktc = np.zeros_like(delta_b_b)
        ktc[delta_b_b <= deltah] = 1
        miu_b_b_norm = sub_traction_coefficient(
            ktc, u_b_b_a_norm, ur_b_b_a_norm, 0, 0, 0, 0, 0, 3
        )
        """
    ###########################################################################
    #                           Force and moment                              #
    ###########################################################################
    f_b_b_I_0 = miu_b_b_norm * Q_b_b

    F_b_b_I_0 = Q_b_b * e_bg_bg_norm
    F_b_b_I_1 = f_b_b_I_0 * e_u_b_b_a_norm
    F_b_b_I_2 = -F_b_b_I_0 + F_b_b_I_1

    qv_b_b = Q_b_b * u_b_b_a_norm
    sv_b_b = f_b_b_I_0 * u_b_b_a_norm
    ###########################################################################
    #         Calculate the sum of contact force between ball and ball        # 
    ###########################################################################
    F_b_b_I = np.zeros((n, 3, 1))
    F_b_b_I[0:-1, :, :] = (-np.diff(F_b_b_I_0, n=1, axis=1) +
                           np.diff(F_b_b_I_1, n=1, axis=1))
    F_b_b_I[-1, :, :] = (-F_b_b_I_0[0, :, :] - F_b_b_I_0[-1, :, :] +
                         F_b_b_I_1[0, :, :] - F_b_b_I_1[-1, :, :])

    F_b_b_a, t0, t1 = np.zeros((n,3,1)), np.zeros((3,3)), np.zeros((3,1))
    for i in range(0, n):
        t0[:,:], t1[:,:] = T_I_a[i,:,:], F_b_b_I[i,:,:]
        t2 = np.dot(t0, t1)
        F_b_b_a[i,:,:] = t2

    M_b_b_I_0 = value_1(R_b*e_bg_bg_norm, F_b_b_I_2)

    M_b_b_I = np.zeros((n, 3, 1))
    M_b_b_I[0:-1, :, :] = np.diff(M_b_b_I_0, n=1, axis=1)
    M_b_b_I[-1, :, :] = M_b_b_I_0[0, :, :] - M_b_b_I_0[-1, :, :]

    M_b_b_b = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 1))

    for i in range(n):
        t0[:, :] = T_I_b[i, :, :]
        t1[:, :], M_b_b_I[i, :, :]
        t2 = np.dot(t0, t1)
        M_b_b_b[i, :, :] = t2
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    Info_bbf = (qv_b_b,
                 # Force*velocity value between ball and other ball.
                sv_b_b,
                 # Traction*slip value between ball and other ball.
                F_b_b_a,
                # Net force of ball from other ball in azimuth frame.
                M_b_b_b,
                # Net moment of ball from other ball in ball frame.
                k_b_b
                # Stiffness between ball and other ball contact.
                )

    F_b_b = (F_b_b_a,
             # Net force of ball from other ball in azimuth frame.
             )

    M_b_b = (M_b_b_b,
             # Net moment of ball from other ball in ball frame.
             )

    return F_b_b, M_b_b, Info_bbf

###############################################################################
#         Calculate load and moment between the outer race and house          #
###############################################################################
# @njit(fastmath=False)
def outer_race_house_force(x, Info_es, Info_brcs, mod_orhf):
    """Solve load and moment between outer race and house.

    Parameters
    ----------
    x: np.darray
        Solution vector.
    Info_es: tuple
        Information of expansion_subcall.
    Info_brcs: tuple
        Information of ball_race_contact_strain.
    Info_brtf: tuple
        Information of traction_force.
    mod_orhf: tuple
        Mode data of outer_race_house_force.

    Returns
    -------
    F_or_h: tuple
        Resultant force between the outer race and house.
    M_or_h: tuple
        Resultant moment between the outer race and house.
    Info_orhf: tuple
        Information of outer_race_house_force.
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    R_o_u, dmpg_o_h, k_ok_h, n = mod_orhf[0::]
    ###########################################################################
    #                               End prepare                               #
    ###########################################################################
    race_u_o_0, hsng_u = Info_es[0], Info_es[4]
    T_o_I, T_I_o = Info_brcs[0:2]

    nk = 360

    R_o_u_eff, R_h_d_eff = R_o_u + race_u_o_0, R_o_u + hsng_u
    ###########################################################################
    #              Contact force between outer race and hourse                #
    ###########################################################################
    alpha_ok_h = np.zeros((nk, 1, 1))
    delta_ok_h = np.zeros((nk, 1, 1))

    for i in range(nk):
        angle = i * 2 * math.pi / nk
        alpha_ok_h[i, 0, 0] = angle
        R_eff_sin = R_o_u_eff * math.sin(angle)
        R_eff_cos = R_o_u_eff * math.cos(angle)
        delta_ok_h[i, 0, 0] = math.sqrt(
            (-R_eff_sin + x[26+12*n])**2 + (R_eff_cos + x[28+12*n])**2
            ) - R_h_d_eff
    ###########################################################################
    #          Relative sliding speed between outer race and hourse           #
    ###########################################################################
    v_o_o = np.zeros((1,3,1))
    v_o_o[0, 0, 0] = (T_I_o[0, 0, 0]*x[25+12*n] +
                      T_I_o[0, 0, 1]*x[27+12*n] +
                      T_I_o[0, 0, 2]*x[29+12*n])
    v_o_o[0, 1, 0] = (T_I_o[0, 1, 0]*x[25+12*n] +
                      T_I_o[0, 1, 1]*x[27+12*n] +
                      T_I_o[0, 1, 2]*x[29+12*n])
    v_o_o[0, 2, 0] = (T_I_o[0, 2, 0]*x[25+12*n] +
                      T_I_o[0, 2, 1]*x[27+12*n] +
                      T_I_o[0, 2, 2]*x[29+12*n])

    v_ok_oa = np.zeros((nk, 3, 1))
    v_ok_oa[:, 0, 0] = v_o_o[0, 0, 0]

    for i in range(nk):
        cos_alpha = math.cos(alpha_ok_h[i, 0, 0])
        sin_alpha = math.sin(alpha_ok_h[i, 0, 0])
        v_ok_oa[i, 1, 0] = (v_o_o[0, 1, 0]*cos_alpha +
                              v_o_o[0, 2, 0]*sin_alpha)
        v_ok_oa[i, 2, 0] = (-v_o_o[0, 1, 0]*sin_alpha +
                              v_o_o[0, 2, 0]*cos_alpha)
    """
    Since the race and house speed are equal, no relative angular velocity
    between them.
    """
    u_ok_h_s = np.zeros((nk, 3, 1))
    # u_ok_h_s[:, 0, 0] = v_ok_oa[:, 0, 0]
    # u_ok_h_s[:, 1, 0] = v_ok_oa[:, 1, 0]
    u_ok_h_s[:, 2, 0] = v_ok_oa[:, 2, 0]
    ###########################################################################
    #                            Force and moment                             #
    ###########################################################################
    q_ok_h = k_ok_h * delta_ok_h
    orhdq = damping_coefficient(dmpg_o_h, 0, m_o, m_o, k_ok_h, q_ok_h, 1, 1)

    F_ok_h_os = np.zeros((nk, 3, 1))
    # F_ok_h_os[:, 0, 0] = 0 * q_ok_h
    # F_ok_h_os[:, 1, 0] = 0 * q_ok_h
    F_ok_h_os[:, 2, 0] = (
        -q_ok_h[:, 0, 0] - orhdq[:, 0, 0] * u_ok_h_s[:, 2, 0]
        )
    F_ok_h_oa = np.copy(F_ok_h_os)
    ###########################################################################
    #     Calculate the sum of contact force between outer race and house     # 
    ###########################################################################
    F_o_h_o = np.zeros((1, 3, 1))
    cos_alpha = np.cos(alpha_ok_h)
    sin_alpha = np.sin(alpha_ok_h)

    for i in range(0, nk):
        F_o_h_o[0, 0, 0] += F_ok_h_oa[i,0,0]
        F_o_h_o[0, 1, 0] += (
            F_ok_h_oa[i, 1, 0]*cos_alpha[i, 0, 0] -
            F_ok_h_oa[i, 2, 0]*sin_alpha[i, 0, 0]
            )
        F_o_h_o[0, 2, 0] += (
            F_ok_h_oa[:, 1, 0]*cos_alpha[:, 0, 0] -
            F_ok_h_oa[i, 2, 0]*sin_alpha[i, 0, 0]
            )

    F_o_h_I = np.zeros((1, 3, 1))
    for i in range(3):
            for j in range(3):
                F_o_h_I[0, i, 0] += T_o_I[0, i, j] * F_o_h_o[0, j, 0]

    F_h_o_i = -F_o_h_I
    """
    Since the race is run at a known speed, and the moment between the race and
    house need to be constrained, so the moment between them are ignored.
    """
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    F_o_h = (F_o_h_I, F_h_o_i)

    M_o_h = (np.zeros((1, 3, 1)), np.zeros((1, 3, 1)))

    return F_o_h, M_o_h

###############################################################################
#                   Calculate fatigue life modified factor                    #
###############################################################################
# @njit(fastmath=False)
def fatigue_life_modified_factor(mtype, ptype, hardness):
    """Solve stle lubrication factor.

    Parameters
    ----------
    mtype: float
        Traction coefficient at zero slip.
    ptype: float
        Rms asperity height, composite roughness.
    hardness: float
        Asperity traction coefficient.

    Returns
    -------
    Info_flmf: float
        Coefficent.
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    mat_name_set_0 = ('AISI 52100', 'AISI M-1', 'AISI M-2', 'AISI M-10',
                      'AISI M-42', 'AISI M-50', 'AISI T-1', 'Halmo', 'WB-49')
    mat_fac_set_0 = (3.0, 0.6, 0.6, 2.0, 0.6, 2.0, 2.0, 2.0, 0.6)
    mat_name_set_1 = ('AMS 5749', 'AMS 5900', 'AISI 440C')
    mat_fac_set_1 = (2.0, 2.0, 0.6)
    mat_name_set_2 = ('M-50 NiL', 'AISI 4720', 'AISI 8620', 'AISI 9310',
                      'CBS 600', 'CBS 1000', 'Vasco X-2')
    mat_fac_set_2 = (4.0, 3.0, 1.5, 2.0, 2.0, 2.0, 2.0)
    mat_name_set_3 = ('Si Nitride')
    mat_fac_set_3 = (6.0,)
    proc_name = ('No Processing', 'AM -Air Melt',
                 'CVD -Carbon Vacuum Degassing', 'VP -Vacuum Processing',
                 'VAR -Vacuum Arc Remelting', 'EFR -Electoflux Remelting',
                 'VAR-VAR -Double Vacuum Arc Remelting',
                 'VIMVAR -Vacuum Induction Melt, Vacuum Arc Remelt')
    proc_fac = (1.0, 1.0, 1.5, 1.5, 3.0, 3.0, 4.5, 6.0)
    ###########################################################################
    #                           STLE materials code                           #
    ###########################################################################
    if mtype == 0:
        stle_mat_name = 'No Life Mod'
        stle_mat_fac = 0.
        stle_proc_name = 'No Life Mod'
        stle_proc_fac = 0.
        stle_hard_fac = 0.
        tallian_mat_name = 'No Life Mod'
        tallian_mat_fac = 0.
        tallian_proc_name = 'No Life Mod'
        tallian_proc_fac = 0.
        tallian_cont_fac = 0.
        #######################################################################
        #                            Store result                             #
        #######################################################################
        Info_flmf = (stle_mat_name,
                     # STLE material name.
                     stle_mat_fac,
                     # STLE material factor.
                     stle_proc_name,
                     # STLE process name.
                     stle_proc_fac,
                     # STLE process factor.
                     stle_hard_fac,
                     # STLE hardness factor.
                     tallian_mat_name,
                     # Tallian material name.
                     tallian_mat_fac,
                     # Tallian material factor.
                     tallian_proc_name,
                     # Tallian process name.
                     tallian_proc_fac,
                     # Tallian process factor.
                     tallian_cont_fac
                     # Tallian contamination factor.
                     )
        return Info_flmf
    else:
        if mtype >= 1 and mtype <= 9:
            stle_mat_name = mat_name_set_0[mtype-1]
            stle_mat_fac = mat_fac_set_0[mtype-1]
        elif mtype == 21 or mtype == 23:
            stle_mat_name = mat_name_set_1[mtype-20]
            stle_mat_fac = mat_fac_set_1[mtype-20]
        elif mtype == 41 or mtype == 47:
            stle_mat_name = mat_name_set_2[mtype-40]
            stle_mat_fac = mat_fac_set_2[mtype-40]
        elif mtype == 51:
            stle_mat_name = mat_name_set_3[0]
            stle_mat_fac = mat_fac_set_3[0]
        #######################################################################
        #                        STLE processing code                         #
        #######################################################################
        stle_proc_name = proc_name[ptype]
        stle_proc_fac = proc_fac[ptype]
        #######################################################################
        #                        STLE hardness factor                         #
        #######################################################################
        if hardness <= 60:
            stle_hard_fac = 1.
        else:
            stle_hard_fac = math.exp(0.1 * (hardness - 60))
            if stle_hard_fac > 2:
                stle_hard_fac = 2.
        #######################################################################
        #                      Tallian materials factor                       #
        #######################################################################
        if mtype == 6 or mtype == 41:
            tallian_mat_name = 'AISI M-50'
            tallian_mat_fac = 2.267
        elif mtype == 43:
            tallian_mat_name = 'AISI 8620'
            tallian_mat_fac = 1.773
        else:
            tallian_mat_name = 'AISI 52100'
            tallian_mat_fac = 1.197
        #######################################################################
        #            Tallian processing and contamination factors             #
        #######################################################################
        if ptype == 7:
            tallian_proc_name =                                               \
                'VIMVAR -Vacuum Induction Melt, Vacuum Arc Remelt'
            tallian_proc_fac = 0.003
            tallian_cont_fac = 0.1
        else:
            tallian_proc_name = 'CVD -Carbon Vacuum Deoxidation'
            tallian_proc_fac = 0.077
            tallian_cont_fac = 1.
        #######################################################################
        #                            Store result                             #
        #######################################################################
        Info_flmf = (stle_mat_name,
                     # STLE material name.
                     stle_mat_fac,
                     # STLE material factor.
                     stle_proc_name,
                     # STLE process name.
                     stle_proc_fac,
                     # STLE process factor.
                     stle_hard_fac,
                     # STLE hardness factor.
                     tallian_mat_name,
                     # Tallian material name.
                     tallian_mat_fac,
                     # Tallian material factor.
                     tallian_proc_name,
                     # Tallian process name.
                     tallian_proc_fac,
                     # Tallian process factor.
                     tallian_cont_fac
                     # Tallian contamination factor.
                     )
        return Info_flmf

###############################################################################
#                    Calculate Tillian modification factor                    #
###############################################################################
# @njit(fastmath=False)
def tallian_modification_factor(film, rms, ds, asptc, slope, em, p, tcm):
    """Solve stle lubrication factor.

    Parameters
    ----------
    film: float
        Traction coefficient at zero slip.
    rms: float
        Rms asperity height, composite roughness.
    ds: float
        Asperity count.
    asptc: float
        Asperity traction coefficient.
    slope: float
        Rms asperity slope (rad), composite slope.
    em: float
        Eq elastic modulus
    p: float
        Contact pressure.
    tcm: float
        Traction coefficent corresponding to max slip.

    Returns
    -------
    phi2a: float
        Coefficent.
    phi3a: float
        Coefficent.
    phi3b: float
        Coefficent.
    phi3f: float
        Coefficent.
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    x_lamda = film / rms
    ###########################################################################
    #                              Compute phi2a                              #
    ###########################################################################
    if x_lamda > 0.4:
        xx = x_lamda - 0.4
    else:
        xx = 0.

    if xx == 0:
        gx = 0.5
    elif xx > 5:
        gx = 0.
    else:
        xx = xx ** 2
        a0 = 7 * math.exp(-0.5 * xx)
        a1 = 16 * math.exp(-xx * (2 - 2 ** 0.5))
        a2 = (7 + 0.25 * math.pi * xx) * math.exp(-xx)
        gx = 0.5 - 0.5 * math.sqrt(1 - (a0 + a1 + a2) / 30)
    phi_2a = ds * gx
    ###########################################################################
    #                     Compute phi3a, phi3b and phi3f                      #
    ###########################################################################
    ss = 0.
    """
    Residual and hoop stress effects are already
    included in basic life estimates

    ss = math.abs(res_stress_r[i] + 0.834 * race_hoop[i]) / p_max_b_r[j, i]
    """
    tb = 0.39 + 0.1 * ss
    # fl_beta, fl_zeta = 1.6, 7.33
    # b_zeta = 11.728, fl_beta * fl_zeta
    phi_3b = tb ** 11.728

    if x_lamda == 0:
        eff_trac = asptc
    elif x_lamda < 4.3:
        f32 = math.exp(-15.5 + 7.5 * (4.3 - x_lamda) ** 0.5)
        # fl_alfa = 40.
        eff_trac = asptc * 0.05 * em * slope * f32 * (40 - 0.9) ** 0.75 / p
        if eff_trac > asptc:
            eff_trac = asptc
    else:
        eff_trac = 0.
    tot_trac = eff_trac + abs(tcm)
    # z_frac = 0.027
    ta = 1.25 * 0.027 + 1.22 * tot_trac + 0.15 * ss
    phi_3a = ta ** 11.728

    tf = 1.25 * 0.027 + 1.22 * asptc + 0.15 * ss
    phi_3f = tf ** 11.728

    return phi_2a, phi_3a, phi_3b, phi_3f

###############################################################################
#                     Calculate STLE modification factor                      #
###############################################################################
# @njit(fastmath=False)
def stle_lubrication_factor(film, rms):
    """Solve stle lubrication factor.

    Parameters
    ----------
    film: float
        Traction coefficient at zero slip.
    rms: float
        Rms asperity height, composite roughness.

    Returns
    -------
    x_lamda: float
        Computed lamda = lub film/composite rms.
    lub_fac: float
        Computed lubrication factor.
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    xl = np.array([0.6, 0.8, 1.0, 1.1, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0])
    fls = np.array([0.19, 0.28, 0.47, 1.00, 1.75, 2.13,
                    2.46, 2.62, 2.83, 2.88, 3.03])
    ###########################################################################
    #       Compute lamda(lub film/composite rms) and lubrication factor      #
    ###########################################################################
    if film > 0:
        x_lamda = film / rms
        if x_lamda <= xl[0]:
            lub_fac = fls[0]
        elif x_lamda >= xl[10]:
            lub_fac = fls[10]
        else:
            xl_len = len(xl)
            i = 0
            while x_lamda > xl[i] and i < xl_len - 1:
                i = i + 1
            m0, m1 = i - 1, i
            lub_fac = (fls[m0] + (fls[m1] - fls[m0])*
                       (x_lamda - xl[m0])/(xl[m1] - xl[m0]))
    else:
        x_lamda = 0.
        lub_fac = fls[0]

    return x_lamda, lub_fac

###############################################################################
#                       Calculate fatigue life constant                       #
###############################################################################
# @njit(fastmath=False)
def fatigue_life_constant(mod_flc):
    """Solve the fatigue life constant.

    Parameters
    ----------
    mod_flc: tuple
        Mode data of fatigue_life_constant.

    Returns
    -------
    Info_flc: tuple
        Information data of fatigue_life_constant.
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    (D_b,
     D_m,
     E_b,
     E_i,
     E_o,
     Shim_thknss_i,
     Shim_thknss_o,
     brg_type,
     f_i,
     f_o,
     hard_coff_b,
     hard_coff_i,
     hard_coff_o,
     k_life_cons,
     mat_fac_type_b,
     mat_fac_type_i,
     mat_fac_type_o,
     mat_type_b,
     max_rs,
     n,
     po_b,
     po_i,
     po_o,
     proc_fac_type_b,
     proc_fac_type_i,
     proc_fac_type_o,
     res_stress_i,
     res_stress_o,
     rms_b,
     rms_i,
     rms_o
     ) = mod_flc[0::]
    ###########################################################################
    #                  Base constants for original LP model                   #
    ###########################################################################
    exp_pt, exp_ln = 3, 4
    ###########################################################################
    #                 Base constants for new fatigue models                   #
    ###########################################################################
    xi_LP, xi_GZ, xi_IH = 0.5, 0.786, 0.66
    eta_LP, eta_GZ, eta_IH = 2., 2., 2.
    zeta_LP, zeta_GZ, zeta_IH = 0.25, 0.3, 0.275
    c_LP, c_GZ, c_IH = 31 / 3 , 31 / 3, 31 / 3
    h_LP, h_IH = 7 / 3, 7 / 3
    e_dis = 10 / 9
    ###########################################################################
    #                Baseline and actual elastic properties                   #
    ###########################################################################
    E_base = 2.0133e11
    po_base = 0.277
    E_M_base = 2 * (1 - po_base ** 2) / E_base
    E_P_base = 1 / E_M_base
    E_R_base = 1.

    E_R = np.zeros(4)
    E_R[0] = E_M_base / ((1 - po_b ** 2) / E_b + (1 - po_o ** 2) / E_o)
    E_R[1] = E_M_base / ((1 - po_b ** 2) / E_b + (1 - po_i ** 2) / E_i)
    E_R[2] = E_R[0]
    E_R[3] = E_R[1]
    ###########################################################################
    #              contact model type based on bearing geometry               #
    ###########################################################################
    k_life_base_type = np.zeros(2)
    if brg_type == 0:
        pass
    elif brg_type == 1:
        pass
        """
        if R_crn_b < 1e6 and Re_cen_len == 0:
            k_life_base_type[0] = 1
            k_life_base_type[1] = 0
        else:
            k_life_base_type[:] = 1
        """

    k_life_mod_fac_r = np.zeros(max_rs)
    k_life_mod_fac_r[0] = mat_fac_type_o
    k_life_mod_fac_r[1] = mat_fac_type_i

    k_life_mod_fac_b = mat_fac_type_b
    ###########################################################################
    #             Defaults for input readable fatigue parameters              #
    ###########################################################################
    if k_life_cons == 0:
        s_prob = 0.9
        reliab_index = 1.

        fac_r_ori = np.zeros(max_rs)
        fac_r_ori[0:2] = 1.

        fac_r_LP = np.zeros(max_rs)
        fac_r_LP[0:2] = 1.

        fac_r_GZ = np.zeros(max_rs)
        fac_r_GZ[0:2] = 1.

        fac_r_IH = np.zeros(max_rs)
        fac_r_IH[0:2] = 1.

        shear_exp_r_LP = np.zeros(max_rs)
        shear_exp_r_LP[0:2] = c_LP

        shear_exp_r_GZ = np.zeros(max_rs)
        shear_exp_r_GZ[0:2] = c_GZ

        shear_exp_r_IH = np.zeros(max_rs)
        shear_exp_r_IH[0:2] = c_IH

        depth_exp_r_LP = np.zeros(max_rs)
        depth_exp_r_LP[0:2] = h_LP

        depth_exp_r_IH = np.zeros(max_rs)
        depth_exp_r_IH[0:2] = h_IH

        shear_exp_b_LP = c_LP
        shear_exp_b_GZ = c_GZ
        shear_exp_b_IH = c_IH

        depth_exp_b_LP = h_LP
        depth_exp_b_IH = h_IH

        wb_dis_brg = e_dis
        wb_dis_b = e_dis
        wb_s = e_dis

        wb_dis_r = np.zeros(max_rs)
        wb_dis_r[:] = e_dis

        fac_b_LP = 1.
        fac_b_GZ = 1.
        fac_b_IH = 1.
    else:
        """
        Need to be changed from upside parameters.
        """
        pass
    ###########################################################################
    #              Default stress capacity constants for races                #
    ###########################################################################
    f_con_pt_r = 2.464006e7 * fac_r_ori
    f_con_ln_r = 1.980630e8 * fac_r_ori

    f_con_pt_r_LP = 1.4598743e9 * fac_r_LP
    f_con_ln_r_LP = (math.pi ** (2 - h_LP) * E_M_base *
                     f_con_pt_r_LP ** (c_LP - h_LP + 2) *
                     0.5 ** (1 - h_LP)) ** (1 / (c_LP - h_LP + 1))

    f_con_pt_r_GZ = 6.4229428e8 * fac_r_GZ
    f_con_ln_r_GZ = (math.pi ** 2 * E_M_base *
                     f_con_pt_r_GZ ** (c_GZ * e_dis + 2) *
                     0.5) ** (1 / (c_GZ * e_dis + 1))

    f_con_pt_r_IH = 1.5524276e9 * fac_r_IH
    f_con_ln_r_IH = (math.pi ** (2 - h_IH) * E_M_base *
                     f_con_pt_r_IH ** (c_IH - h_IH + 2) *
                     0.5 ** (1 - h_IH)) ** (1 / (c_IH - h_IH + 1))
    ###########################################################################
    #                    higher shear stress exponents and                    #
    #     modified stress capacity constants for ceramic rolling elements     #
    #                            (Only for Si3N4)                             #
    ###########################################################################
    if mat_type_b >= 5 and mat_type_b <= 7:
        #######################################################################
        #                      Shear stress exponents                         #
        #######################################################################
        c_sin = 16 * e_dis + h_LP - 2

        shear_exp_b_LP = c_sin
        shear_exp_b_GZ = shear_exp_b_LP
        shear_exp_b_IH = shear_exp_b_LP
        #######################################################################
        #                     Stress capacity constantss                      #
        #######################################################################
        f_con_pt_b_LP = f_con_pt_r_LP[0] * 1.00912067 * fac_b_LP
        f_con_ln_b_LP = (math.pi ** (2 - h_LP) * E_M_base *
                         f_con_pt_b_LP ** (c_LP - h_LP + 2) /
                         0.5 ** (1 - h_LP) ** (1 / (c_LP - h_LP + 1)))

        f_con_pt_b_GZ = f_con_pt_r_GZ[0] * 1.44866477 * fac_b_GZ
        f_con_ln_b_GZ = (math.pi ** 2 * E_M_base *
                         f_con_pt_b_GZ ** (c_GZ * e_dis + 2) /
                         0.5) ** (1 / (c_GZ * e_dis + 1))

        f_con_pt_b_IH = f_con_pt_r_IH[0] * 1.21120732 * fac_b_IH
        f_con_ln_b_IH = (math.pi ** (2 - h_IH) * E_M_base *
                         f_con_pt_b_IH ** (c_IH - h_IH + 2) /
                         0.5 ** (1 - h_IH) ** (1 / (c_IH - h_IH + 1)))
    else:
        f_con_pt_b_LP = f_con_pt_r_LP[0]
        f_con_ln_b_LP = f_con_ln_r_LP[0]

        f_con_pt_b_GZ = f_con_pt_r_GZ[0]
        f_con_ln_b_GZ = f_con_ln_r_GZ[0]

        f_con_pt_b_IH = f_con_pt_r_IH[0]
        f_con_ln_b_IH = f_con_ln_r_IH[0]
    ###########################################################################
    #               Constant factors in load capacity equations               #
    ###########################################################################
    reliab_index = math.log(1 / s_prob) / math.log(1 / 0.9)

    c_kapa_r_LP = (eta_LP * zeta_LP ** shear_exp_r_LP *
                   xi_LP ** (1 - depth_exp_r_LP) / reliab_index)
    c_kapa_r_GZ = (eta_GZ * zeta_GZ ** (shear_exp_r_GZ * wb_dis_r) *
                   xi_GZ / reliab_index)
    c_kapa_r_IH = (eta_IH * zeta_IH ** shear_exp_r_IH *
                   xi_IH ** (1 - depth_exp_r_IH) / reliab_index)

    c_kapa_b_LP = (eta_LP * zeta_LP ** shear_exp_b_LP *
                   xi_LP ** (1 - depth_exp_b_LP) / reliab_index)
    c_kapa_b_GZ = (eta_GZ * zeta_GZ ** (shear_exp_b_GZ * wb_dis_b) *
                   xi_GZ / reliab_index)
    c_kapa_b_IH = (eta_IH * zeta_IH ** shear_exp_b_IH *
                   xi_IH ** (1 - depth_exp_b_IH) /reliab_index)
    ###########################################################################
    #                           Constants for races                           #
    ###########################################################################
    wb_dis_brg_inv = 1 / wb_dis_brg
    wb_dis_r_inv = 1 / wb_dis_r
    wb_s_inv = 1 / wb_s

    c_kapa_pt_r_LP = (
        1 / c_kapa_r_LP ** (1 / (shear_exp_r_LP - depth_exp_r_LP + 2))
    )
    c_kapa_ln_r_LP = (
        1 / c_kapa_r_LP ** (1 / (shear_exp_r_LP - depth_exp_r_LP + 1))
    )

    c_kapa_pt_r_GZ = 1 / c_kapa_r_GZ ** (1 / (shear_exp_r_GZ * wb_dis_r + 2))
    c_kapa_ln_r_GZ = 1 / c_kapa_r_GZ ** (1 / (shear_exp_r_GZ * wb_dis_r + 1))

    c_kapa_pt_r_IH = (
        1 / c_kapa_r_IH ** (1 / (shear_exp_r_IH - depth_exp_r_IH + 2))
    )
    c_kapa_ln_r_IH = (
        1 / c_kapa_r_IH ** (1 / (shear_exp_r_IH - depth_exp_r_IH + 1))
    )
    ###########################################################################
    #                     Constants for rolling elements                      #
    ###########################################################################
    wb_dis_b_inv = 1 / wb_dis_b

    c_kapa_pt_b_LP = (
        1 / c_kapa_b_LP ** (1 / (shear_exp_b_LP - depth_exp_b_LP + 2))
    )
    c_kapa_ln_b_LP = (
        1 / c_kapa_b_LP ** (1 / (shear_exp_b_LP - depth_exp_b_LP + 1))
    )

    c_kapa_pt_b_GZ = 1 / c_kapa_b_GZ ** (1 / (shear_exp_b_GZ * wb_dis_b + 2))
    c_kapa_ln_b_GZ = 1 / c_kapa_b_GZ ** (1 / (shear_exp_b_GZ * wb_dis_b + 1))

    c_kapa_pt_b_IH = (
        1 / c_kapa_b_IH ** (1 / (shear_exp_b_IH - depth_exp_b_IH + 2))
    )
    c_kapa_ln_b_IH = (
        1 / c_kapa_b_IH ** (1 / (shear_exp_b_IH - depth_exp_b_IH + 1))
    )
    ###########################################################################
    #                     Constants for original LP model                     #
    ###########################################################################
    if k_life_base_type[0] == 0:
        #######################################################################
        #                            Point contact                            #
        #######################################################################
        # e_l_r = 1.8
        f_con_r = f_con_pt_r

        q_exp_r = np.zeros(max_rs)
        q_exp_r[:] = exp_pt

        q_exp_r_inv = np.zeros(max_rs)
        q_exp_r_inv[:] = 1 / exp_pt
    else:
        #######################################################################
        #                            Line contact                             #
        #######################################################################
        # e_l_r = 50 / 27
        f_con_r = f_con_ln_r

        q_exp_r = np.zeros(max_rs)
        q_exp_r[:] = exp_ln

        q_exp_r_inv = np.zeros(max_rs)
        q_exp_r_inv[:] = 1 / exp_ln
    ###########################################################################
    #                               Set arrays                                #
    ###########################################################################
    (
        e_l_r_LP, e_l_r_GZ, e_l_r_IH,
        e_g_r_LP, e_g_r_GZ, e_g_r_IH,
        f_con_r_LP, f_con_r_GZ, f_con_r_IH,
        p_exp_r_LP, p_exp_r_GZ, p_exp_r_IH,
        q_exp_r_LP, q_exp_r_GZ, q_exp_r_IH,
        u_exp_r_LP, u_exp_r_GZ, u_exp_r_IH,
        g_exp_r_LP, g_exp_r_GZ, g_exp_r_IH,
        g0_exp_r_LP, g0_exp_r_GZ, g0_exp_r_IH,
        g1_exp_r_LP, g1_exp_r_GZ, g1_exp_r_IH,
        g2_exp_r_LP, g2_exp_r_GZ, g2_exp_r_IH,
        pr_exp_r_LP, pr_exp_r_GZ, pr_exp_r_IH
        ) = (
            np.zeros(max_rs), np.zeros(max_rs), np.zeros(max_rs),
            np.zeros(max_rs), np.zeros(max_rs), np.zeros(max_rs),
            np.zeros(max_rs), np.zeros(max_rs), np.zeros(max_rs),
            np.zeros(max_rs), np.zeros(max_rs), np.zeros(max_rs),
            np.zeros(max_rs), np.zeros(max_rs), np.zeros(max_rs),
            np.zeros(max_rs), np.zeros(max_rs), np.zeros(max_rs),
            np.zeros(max_rs), np.zeros(max_rs), np.zeros(max_rs),
            np.zeros(max_rs), np.zeros(max_rs), np.zeros(max_rs),
            np.zeros(max_rs), np.zeros(max_rs), np.zeros(max_rs),
            np.zeros(max_rs), np.zeros(max_rs), np.zeros(max_rs),
            np.zeros(max_rs), np.zeros(max_rs), np.zeros(max_rs)
            )
    ###########################################################################
    #                              Point contact                              #
    ###########################################################################
    if k_life_base_type[1] == 0:
        #######################################################################
        #                 Current LP model race constants                     #
        #######################################################################
        coff_r_LP = shear_exp_r_LP - depth_exp_r_LP + 2
        e_l_r_LP[0:2] = (
            2 * shear_exp_r_LP[0:2] - depth_exp_r_LP[0:2] + 1
        ) / coff_r_LP[0:2]

        f_con_r_LP[0:2] = f_con_pt_r_LP[0:2]

        p_exp_r_LP[0:2] = coff_r_LP[0:2] / wb_dis_r[0:2]
        q_exp_r_LP[0:2] = coff_r_LP[0:2] / (3 * wb_dis_r[0:2])

        g0_exp_r_LP[0:2] = 3 - depth_exp_r_LP[0:2]
        g1_exp_r_LP[0:2] = 3 - 2 * depth_exp_r_LP[0:2]
        g2_exp_r_LP[0:2] = depth_exp_r_LP[0:2] - 2
        g_exp_r_LP[0:2] = -1 / coff_r_LP[0:2]

        e_g_r_LP[0:2] = 3 - depth_exp_r_LP[0:2]

        pr_exp_r_LP[0:2] = (2 - depth_exp_r_LP[0:2]) / coff_r_LP[0:2]

        u_exp_r_LP[0:2] = -wb_dis_r[0:2] / coff_r_LP[0:2]
        #######################################################################
        #                 Current GZ model race constants                     #
        #######################################################################
        e_l_r_GZ[0:2] = 2 * shear_exp_r_GZ[0:2] * wb_dis_r[0:2] + 1

        f_con_r_GZ[0:2] = f_con_pt_r_GZ[0:2]

        p_exp_r_GZ[0:2] = (
            shear_exp_r_GZ[0:2] * wb_dis_r[0:2] + 2
        ) / wb_dis_r[0:2]
        q_exp_r_GZ[0:2] = (
            2 + shear_exp_r_GZ[0:2] * wb_dis_r[0:2]
        ) / (3 * wb_dis_r[0:2])

        g0_exp_r_GZ[0:2] = 3.
        g1_exp_r_GZ[0:2] = 3.
        g2_exp_r_GZ[0:2] = -2.
        g_exp_r_GZ[0:2] = -1 / (shear_exp_r_GZ[0:2] * wb_dis_r[0:2] + 2)

        e_g_r_GZ[0:2] = 3.

        pr_exp_r_GZ[0:2] = 2 / (shear_exp_r_GZ[0:2] * wb_dis_r[0:2] + 2)

        u_exp_r_GZ[0:2] = -wb_dis_r[0:2] / (
            shear_exp_r_GZ[0:2] * wb_dis_r[0:2] + 2
        )
        #######################################################################
        #                 Current IH model race constants                     #
        #######################################################################
        coff_r_IH = shear_exp_r_IH - depth_exp_r_IH + 2
        e_l_r_IH[0:2] = (
            2 * shear_exp_r_IH[0:2] - depth_exp_r_IH[0:2] + 1
        ) / coff_r_IH[0:2]

        f_con_r_IH[0:2] = f_con_pt_r_IH[0:2]

        p_exp_r_IH[0:2] = coff_r_IH[0:2] / wb_dis_r[0:2]
        q_exp_r_IH[0:2] = coff_r_IH[0:2] / (3 * wb_dis_r[0:2])

        g0_exp_r_IH[0:2] = 3 - depth_exp_r_IH[0:2]
        g1_exp_r_IH[0:2] = 3 - 2 * depth_exp_r_IH[0:2]
        g2_exp_r_IH[0:2] = depth_exp_r_IH[0:2] - 2
        g_exp_r_IH[0:2] = -1 / coff_r_IH[0:2]

        e_g_r_IH[0:2] = 3 - depth_exp_r_IH[0:2]

        pr_exp_r_IH[0:2] = (2 - depth_exp_r_IH[0:2]) / coff_r_IH[0:2]

        u_exp_r_IH[0:2] = -wb_dis_r[0:2] / coff_r_IH[0:2]
        #######################################################################
        #      Current LP model point contact rolling element constants       #
        #######################################################################
        coff_b_LP = shear_exp_b_LP - depth_exp_b_LP + 2
        # e_l_b_LP = (2 * shear_exp_b_LP - depth_exp_b_LP + 1) / coff_b_LP

        f_con_b_LP = f_con_pt_b_LP

        p_exp_b_LP = coff_b_LP / wb_dis_b
        # q_exp_b_LP = coff_b_LP / (3 * wb_dis_b)

        g0_exp_b_LP = 3 - depth_exp_b_LP
        g1_exp_b_LP = 3 - 2 * depth_exp_b_LP
        g2_exp_b_LP = depth_exp_b_LP - 2
        g_exp_b_LP = -1 / coff_b_LP

        # e_g_b_LP = 3 - depth_exp_b_LP

        pr_exp_b_LP = (2 - depth_exp_b_LP) / coff_b_LP

        u_exp_b_LP = -wb_dis_b / coff_b_LP
        #######################################################################
        #      Current GZ model point contact rolling element constants       #
        #######################################################################
        # e_l_b_GZ = 2 * shear_exp_b_GZ * wb_dis_b + 1

        f_con_b_GZ = f_con_pt_b_GZ

        p_exp_b_GZ = (shear_exp_b_GZ * wb_dis_b + 2) / wb_dis_b
        # q_exp_b_GZ = (2 + shear_exp_b_GZ * wb_dis_b) / (3 * wb_dis_b)

        g0_exp_b_GZ = 3.
        # g1_exp_b_GZ = 3.
        g2_exp_b_GZ = -2.
        g_exp_b_GZ = -1 / (shear_exp_b_GZ * wb_dis_b + 2)

        # e_g_b_GZ = 3.

        pr_exp_b_GZ = 2 / (shear_exp_b_GZ * wb_dis_b + 2)

        u_exp_b_GZ = -wb_dis_b / (shear_exp_b_GZ * wb_dis_b + 2)
        #######################################################################
        #      Current IH model point contact rolling element constants       #
        #######################################################################
        coff_b_IH = shear_exp_b_IH - depth_exp_b_IH + 2
        # e_l_b_IH = (2 * shear_exp_b_IH - depth_exp_b_IH + 1) / coff_b_IH

        f_con_b_IH = f_con_pt_b_IH

        p_exp_b_IH = coff_b_IH / wb_dis_b
        # q_exp_b_IH = coff_b_IH / (3 * wb_dis_b)

        g0_exp_b_IH = 3 - depth_exp_b_IH
        g1_exp_b_IH = 3 - 2 * depth_exp_b_IH
        g2_exp_b_IH = depth_exp_b_IH - 2
        g_exp_b_IH = -1 / coff_b_IH

        # e_g_b_IH = 3 - depth_exp_b_IH

        pr_exp_b_IH = (2 - depth_exp_b_IH) / coff_b_IH

        u_exp_b_IH = -wb_dis_b / coff_b_IH
    ###########################################################################
    #                              Line contact                               #
    ###########################################################################
    else:
        #######################################################################
        #                 Current LP model race constants                     #
        #######################################################################
        coff_r_LP = shear_exp_r_LP - depth_exp_r_LP + 1
        e_l_r_LP[0:2] = (
            2 * shear_exp_r_LP[0:2] - depth_exp_r_LP[0:2] - 1
        ) / coff_r_LP[0:2]

        f_con_r_LP[0:2] = f_con_ln_r_LP[0:2]

        p_exp_r_LP[0:2] = coff_r_LP[0:2] / wb_dis_r[0:2]
        q_exp_r_LP[0:2] = coff_r_LP[0:2] / (2 * wb_dis_r[0:2])

        g0_exp_r_LP[0:2] = 1.
        g1_exp_r_LP[0:2] = 1.
        g2_exp_r_LP[0:2] = depth_exp_r_LP[0:2] - 1
        g_exp_r_LP[0:2] = -1 / coff_r_LP[0:2]

        e_g_r_LP[0:2] = 3 - depth_exp_r_LP[0:2]

        pr_exp_r_LP[0:2] = (1 - depth_exp_r_LP[0:2]) / coff_r_LP[0:2]

        u_exp_r_LP[0:2] = -wb_dis_r[0:2] / coff_r_LP[0:2]
        #######################################################################
        #                 Current GZ model race constants                     #
        #######################################################################
        coff_r_GZ = shear_exp_r_GZ * wb_dis_r + 1
        e_l_r_GZ[0:2] = (
            2 * shear_exp_r_GZ[0:2] * wb_dis_r[0:2] - 1
        ) / coff_r_GZ[0:2]

        f_con_r_GZ[0:2] = f_con_ln_r_GZ[0:2]

        p_exp_r_GZ[0:2] = coff_r_GZ[0:2] / wb_dis_r[0:2]
        q_exp_r_GZ[0:2] = coff_r_GZ[0:2] / (2 * wb_dis_r[0:2])

        g0_exp_r_GZ[0:2] = 1.
        g1_exp_r_GZ[0:2] = 1.
        g2_exp_r_GZ[0:2] = 1.
        g_exp_r_GZ[0:2] = -1 / coff_r_GZ[0:2]
        
        e_g_r_GZ[0:2] = 3.

        pr_exp_r_GZ[0:2] = 1 / coff_r_GZ[0:2]

        u_exp_r_GZ[0:2] = -wb_dis_r[0:2] / coff_r_GZ[0:2]
        #######################################################################
        #                 Current IH model race constants                     #
        #######################################################################
        coff_r_IH = shear_exp_r_IH - depth_exp_r_IH + 1
        e_l_r_IH[0:2] =  (
            2 * shear_exp_r_IH[0:2] - depth_exp_r_IH[0:2] - 1
        ) / coff_r_IH[0:2]

        f_con_r_IH[0:2] = f_con_ln_r_IH[0:2]

        p_exp_r_IH[0:2] = coff_r_IH[0:2] / wb_dis_r[0:2]
        q_exp_r_IH[0:2] = coff_r_IH[0:2] / (2 * wb_dis_r[0:2])

        g0_exp_r_IH[0:2] = 1.
        g1_exp_r_IH[0:2] = 1.
        g2_exp_r_IH[0:2] = depth_exp_r_IH[0:2] - 1
        g_exp_r_IH[0:2] = -1 / coff_r_IH[0:2]

        e_g_r_IH[0:2] = 3 - depth_exp_r_IH[0:2]

        pr_exp_r_IH[0:2] = (1 - depth_exp_r_IH[0:2]) / coff_r_IH[0:2]

        u_exp_r_IH[0:2] = -wb_dis_r[0:2] / coff_r_IH[0:2]
        #######################################################################
        #      Current LP model point contact rolling element constants       #
        #######################################################################
        coff_b_LP = shear_exp_b_LP - depth_exp_b_LP + 1
        # e_l_b_LP = (2 * shear_exp_b_LP - depth_exp_b_LP - 1) / coff_b_LP

        f_con_b_LP = f_con_ln_b_LP

        p_exp_b_LP = coff_b_LP / wb_dis_b
        # q_exp_b_LP = coff_b_LP / (2 * wb_dis_b)

        g0_exp_b_LP = 1.
        g1_exp_b_LP = 1.
        g2_exp_b_LP = depth_exp_b_LP - 1
        g_exp_b_LP = -1 / coff_b_LP

        # e_g_b_LP = 3 - depth_exp_b_LP

        pr_exp_b_LP = (1 - depth_exp_b_LP) / coff_b_LP

        u_exp_b_LP = -wb_dis_b / coff_b_LP
        #######################################################################
        #      Current GZ model point contact rolling element constants       #
        #######################################################################
        coff_b_GZ = shear_exp_b_GZ * wb_dis_b + 1
        # e_l_b_GZ = (2 * shear_exp_b_GZ * wb_dis_b - 1) / coff_b_GZ

        f_con_b_GZ = f_con_ln_b_GZ

        p_exp_b_GZ = coff_b_GZ / wb_dis_b
        # q_exp_b_GZ = coff_b_GZ / (2 * wb_dis_b)

        g0_exp_b_GZ = 1.
        # g1_exp_b_GZ = 1.
        g2_exp_b_GZ = 1.
        g_exp_b_GZ = -1 / coff_b_GZ

        # e_g_b_GZ = 3.

        pr_exp_b_GZ = 1 / coff_b_GZ

        u_exp_b_GZ = -wb_dis_b / coff_b_GZ
        #######################################################################
        #      Current IH model point contact rolling element constants       #
        #######################################################################
        coff_b_IH = shear_exp_b_IH - depth_exp_b_IH + 1
        # e_l_b_IH = (2 * shear_exp_b_IH - depth_exp_b_IH - 1) / coff_b_IH

        f_con_b_IH = f_con_ln_b_IH

        p_exp_b_IH = coff_b_IH / wb_dis_b
        # q_exp_b_IH = coff_b_IH / (2 * wb_dis_b)

        g0_exp_b_IH = 1.
        g1_exp_b_IH = 1.
        g2_exp_b_IH = depth_exp_b_IH - 1
        g_exp_b_IH = -1 / coff_b_IH

        # e_g_b_IH = 3 - depth_exp_b_IH

        pr_exp_b_IH = (1 - depth_exp_b_IH) / coff_b_IH

        u_exp_b_IH = -wb_dis_b / coff_b_IH
    ###########################################################################
    #                 Averaging exponents for load and length                 #
    ###########################################################################
    q_exp_load_r = np.zeros((2, 2))
    q_exp_load_r[0, 0] = q_exp_r[0] * wb_s
    q_exp_load_r[1, 0] = q_exp_r[1] * wb_s
    q_exp_load_r[0, 1] = q_exp_load_r[0, 0]
    q_exp_load_r[1, 1] = q_exp_load_r[1, 0]

    q_exp_load_r_inv = np.zeros((2, 2))
    q_exp_load_r_inv[:, :] = 1 / q_exp_load_r

    q_exp_len_r = np.zeros((2, 2))
    q_exp_len_r[0, 0] = (q_exp_load_r[0, 0] - 1) / q_exp_load_r[0, 0]
    q_exp_len_r[1, 0] = (q_exp_load_r[0, 1] - 1) / q_exp_load_r[0, 1]
    q_exp_len_r[0, 1] = (q_exp_load_r[1, 0] - 1) / q_exp_load_r[1, 0]
    q_exp_len_r[1, 1] = (q_exp_load_r[1, 1] - 1) / q_exp_load_r[1, 1]

    q_exp_load_r_LP = np.zeros((2, 2))
    q_exp_load_r_LP[0, 0] = q_exp_r_LP[0] * wb_dis_r[0]
    q_exp_load_r_LP[1, 0] = q_exp_r_LP[1] * wb_dis_r[1]
    q_exp_load_r_LP[0, 1] = q_exp_load_r_LP[0, 0]
    q_exp_load_r_LP[1, 1] = q_exp_load_r_LP[1, 0]

    q_exp_load_r_LP_inv = np.zeros((2, 2))
    q_exp_load_r_LP_inv[:, :] = 1 / q_exp_load_r_LP

    q_exp_len_r_LP = np.zeros((2, 2))
    q_exp_len_r_LP[0, 0] = (q_exp_load_r_LP[0, 0] - 1) / q_exp_load_r_LP[0, 0]
    q_exp_len_r_LP[1, 0] = (q_exp_load_r_LP[0, 1] - 1) / q_exp_load_r_LP[0, 1]
    q_exp_len_r_LP[0, 1] = (q_exp_load_r_LP[1, 0] - 1) / q_exp_load_r_LP[1, 0]
    q_exp_len_r_LP[1, 1] = (q_exp_load_r_LP[1, 1] - 1) / q_exp_load_r_LP[1, 1]

    q_exp_load_r_GZ = np.zeros((2, 2))
    q_exp_load_r_GZ[0, 0] = q_exp_r_LP[0] * wb_dis_r[0]
    q_exp_load_r_GZ[1, 0] = q_exp_r_LP[1] * wb_dis_r[1]
    q_exp_load_r_GZ[0, 1] = q_exp_load_r_GZ[0, 0]
    q_exp_load_r_GZ[1, 1] = q_exp_load_r_GZ[1, 0]

    q_exp_load_r_GZ_inv = np.zeros((2, 2))
    q_exp_load_r_GZ_inv[:, :] = 1 / q_exp_load_r_GZ

    q_exp_len_r_GZ = np.zeros((2, 2))
    q_exp_len_r_GZ[0, 0] = (q_exp_load_r_GZ[0, 0] - 1) / q_exp_load_r_GZ[0, 0]
    q_exp_len_r_GZ[1, 0] = (q_exp_load_r_GZ[0, 1] - 1) / q_exp_load_r_GZ[0, 1]
    q_exp_len_r_GZ[0, 1] = (q_exp_load_r_GZ[1, 0] - 1) / q_exp_load_r_GZ[1, 0]
    q_exp_len_r_GZ[1, 1] = (q_exp_load_r_GZ[1, 1] - 1) / q_exp_load_r_GZ[1, 1]
    ###########################################################################
    #       Setup outer race Tallian and STLE Life Modification Factors       #
    ###########################################################################
    if k_life_mod_fac_r[0] != -1:
        shear_limt_o = 0.
        asp_trac_o = 0.1
        asp_ht_o = math.sqrt(rms_b ** 2 + rms_o ** 2) * 1e3
        asp_slope_o = asp_ht_o ** (1 - 0.2607) / 0.02683
        #######################################################################
        #            call procedures to set up outer race factors             #
        #######################################################################
        con_flmf_o = fatigue_life_modified_factor(
            mat_fac_type_o, proc_fac_type_o, hard_coff_o
        )

        (
            stle_mat_name_o, stle_mat_fac_o,
            stle_proc_name_o, stle_proc_fac_o,
            stle_hard_fac_o, tallian_mat_name_o,
            tallian_mat_fac_o, tallian_proc_name_o,
            tallian_proc_fac_o, tallian_cont_fac_o
        ) = con_flmf_o[:]
    else:
        pass
        # stle_mat_name_o = 'User Defined'
        # stle_mat_fac_o = 'User Defined'
        # stle_proc_name_o = 'User Defined'
        # stle_proc_fac_o = 'User Defined'
        # stle_hard_fac_o = 'User Defined'
        # tallian_mat_name_o = 'User Defined'
        # tallian_mat_fac_o = 'User Defined'
        # tallian_proc_name_o = 'User Defined'
        # tallian_proc_fac_o = 'User Defined'
        # tallian_cont_fac_o = 'User Defined'
    prod_fac_o = stle_hard_fac_o * stle_mat_fac_o * stle_proc_fac_o
    ###########################################################################
    #       Setup inner race Tallian and STLE Life Modification Factors       #
    ###########################################################################
    if k_life_mod_fac_r[1] != -1:
        shear_limt_i = 0.
        asp_trac_i = 0.1
        asp_ht_i = math.sqrt(rms_b ** 2 + rms_i ** 2) * 1e3
        asp_slope_i = asp_ht_i ** (1 - 0.2607) / 0.02683
        #######################################################################
        #            call procedures to set up inner race factors             #
        #######################################################################
        con_flmf_i =  fatigue_life_modified_factor(
            mat_fac_type_i, proc_fac_type_i, hard_coff_i
        )

        (
            stle_mat_name_i, stle_mat_fac_i,
            stle_proc_name_i, stle_proc_fac_i,
            stle_hard_fac_i, tallian_mat_name_i,
            tallian_mat_fac_i, tallian_proc_name_i,
            tallian_proc_fac_i, tallian_cont_fac_i
        ) = con_flmf_i[:]
    else:
        pass
        # stle_mat_name_i = 'User Defined'
        # stle_mat_fac_i = 'User Defined'
        # stle_proc_name_i = 'User Defined'
        # stle_proc_fac_i = 'User Defined'
        # stle_hard_fac_i = 'User Defined'
        # tallian_mat_name_i = 'User Defined'
        # tallian_mat_fac_i = 'User Defined'
        # tallian_proc_name_i = 'User Defined'
        # tallian_proc_fac_i = 'User Defined'
        # tallian_cont_fac_i = 'User Defined'
    prod_fac_i = stle_hard_fac_i * stle_mat_fac_i * stle_proc_fac_i
    ###########################################################################
    #       Setup outer race Tallian and STLE Life Modification Factors       #
    ###########################################################################
    if k_life_mod_fac_b != -1:
        #######################################################################
        #            call procedures to set up outer race factors             #
        #######################################################################
        con_flmf_b = fatigue_life_modified_factor(
            mat_fac_type_b, proc_fac_type_b, hard_coff_b
        )

        (
            stle_mat_name_b, stle_mat_fac_b,
            stle_proc_name_b, stle_proc_fac_b,
            stle_hard_fac_b, tallian_mat_name_b,
            tallian_mat_fac_b, tallian_proc_name_b,
            tallian_proc_fac_b, tallian_cont_fac_b
        ) = con_flmf_b[:]
    else:
        pass
        # stle_mat_name_b = 'User Defined'
        # stle_mat_fac_b = 'User Defined'
        # stle_proc_name_b = 'User Defined'
        # stle_proc_fac_b = 'User Defined'
        # stle_hard_fac_b = 'User Defined'
        # tallian_mat_name_b = 'User Defined'
        # tallian_mat_fac_b = 'User Defined'
        # tallian_proc_name_b = 'User Defined'
        # tallian_proc_fac_b = 'User Defined'
        # tallian_cont_fac_b = 'User Defined'
    proc_fac_b = tallian_proc_fac_b
    prod_fac_b = stle_hard_fac_b * stle_mat_fac_b * stle_proc_fac_b
    ###########################################################################
    #           Set data for secondary race surfaces for split race           #
    ###########################################################################
    Shim_thknss_r = (Shim_thknss_o, Shim_thknss_i)

    tallian_mat_fac_r = np.zeros(max_rs)
    tallian_mat_fac_r[0] = con_flmf_o[6]
    tallian_mat_fac_r[1] = con_flmf_i[6]

    shear_limt_r = np.zeros(max_rs)
    shear_limt_r[0] = shear_limt_o
    shear_limt_r[1] = shear_limt_i

    asp_ht_b_r = np.zeros(max_rs)
    asp_ht_b_r[0] = rms_o
    asp_ht_b_r[1] = rms_i

    asp_trac_b_r = np.zeros(max_rs)
    asp_trac_b_r[0] = asp_trac_o
    asp_trac_b_r[1] = asp_trac_i

    asp_slope_b_r = np.zeros(max_rs)
    asp_slope_b_r[0] = asp_slope_o
    asp_slope_b_r[1] = asp_slope_i

    sig_theta_r = np.zeros(max_rs)
    sig_theta_r[0] = rms_o / asp_slope_o * 1e3
    sig_theta_r[1] = rms_i / asp_slope_i * 1e3

    cont_fac_r = np.zeros(max_rs)
    cont_fac_r[0] = con_flmf_o[9]
    cont_fac_r[1] = con_flmf_i[9]

    proc_fac_r = np.zeros(max_rs)
    proc_fac_r[0] = tallian_proc_fac_o
    proc_fac_r[1] = tallian_proc_fac_i

    prod_fac_r = np.zeros(max_rs)
    prod_fac_r[0] = prod_fac_o
    prod_fac_r[1] = prod_fac_i

    taur_r_GZ = np.zeros(max_rs)
    taur_r_GZ[0] = 0.5 * res_stress_o
    taur_r_GZ[1] = 0.5 * res_stress_i

    taur_r_IH = np.zeros(max_rs)
    taur_r_IH[0] = 0.4714 * res_stress_o
    taur_r_IH[1] = 0.4714 * res_stress_i

    for i in range(2):
        if max_rs > 2:
            k_life_mod_fac_r[i+2] = k_life_mod_fac_r[i]
            ###################################################################
            #                     Fatigue life parameters                     #
            ###################################################################
            f_con_r[i+2] = f_con_r[i]
            q_exp_r[i+2] = q_exp_r[i]
            q_exp_r_inv[i+2] = q_exp_r_inv[i]
            f_con_r_LP[i+2] = f_con_r_LP[i]
            wb_dis_r[i+2] = wb_dis_r[i]
            wb_dis_r_inv[i+2] = wb_dis_r_inv[i]
            ###################################################################
            #                            LP model                             #
            ###################################################################
            f_con_r_LP[i+2] = f_con_r_LP[i]
            e_l_r_LP[i+2] = e_l_r_LP[i]
            f_con_r_LP[i+2] = f_con_r_LP[i]
            shear_exp_r_LP[i+2] = shear_exp_r_LP[i]
            depth_exp_r_LP[i+2] = depth_exp_r_LP[i]
            q_exp_r_LP[i+2] = q_exp_r_LP[i]
            g0_exp_r_LP[i+2] = g0_exp_r_LP[i]
            g1_exp_r_LP[i+2] = g1_exp_r_LP[i]
            g2_exp_r_LP[i+2] = g2_exp_r_LP[i]
            p_exp_r_LP[i+2] = p_exp_r_LP[i]
            u_exp_r_LP[i+2] = u_exp_r_LP[i]
            c_kapa_pt_r_LP[i+2] = c_kapa_pt_r_LP[i]
            c_kapa_ln_r_LP[i+2] = c_kapa_ln_r_LP[i]
            ###################################################################
            #                            GZ model                             #
            ###################################################################
            f_con_r_GZ[i+2] = f_con_r_GZ[i]
            e_l_r_GZ[i+2] = e_l_r_GZ[i]
            f_con_r_GZ[i+2] = f_con_r_GZ[i]
            shear_exp_r_GZ[i+2] = shear_exp_r_GZ[i]
            q_exp_r_GZ[i+2] = q_exp_r_GZ[i]
            g0_exp_r_GZ[i+2] = g0_exp_r_GZ[i]
            g1_exp_r_GZ[i+2] = g1_exp_r_GZ[i]
            g2_exp_r_GZ[i+2] = g2_exp_r_GZ[i]
            p_exp_r_GZ[i+2] = p_exp_r_GZ[i]
            u_exp_r_GZ[i+2] = u_exp_r_GZ[i]
            c_kapa_pt_r_GZ[i+2] = c_kapa_pt_r_GZ[i]
            c_kapa_ln_r_GZ[i+2] = c_kapa_ln_r_GZ[i]
            taur_r_GZ[i+2] = taur_r_GZ[i]
            ###################################################################
            #                            IH model                             #
            ###################################################################
            f_con_r_IH[i+2] = f_con_r_IH[i]
            e_l_r_IH[i+2] = e_l_r_IH[i]
            f_con_r_IH[i+2] = f_con_r_IH[i]
            shear_exp_r_IH[i+2] = shear_exp_r_IH[i]
            depth_exp_r_IH[i+2] = depth_exp_r_IH[i]
            q_exp_r_IH[i+2] = q_exp_r_IH[i]
            g0_exp_r_IH[i+2] = g0_exp_r_IH[i]
            g1_exp_r_IH[i+2] = g1_exp_r_IH[i]
            g2_exp_r_IH[i+2] = g2_exp_r_IH[i]
            p_exp_r_IH[i+2] = p_exp_r_IH[i]
            u_exp_r_IH[i+2] = u_exp_r_IH[i]
            c_kapa_pt_r_IH[i+2] = c_kapa_pt_r_IH[i]
            c_kapa_ln_r_IH[i+2] = c_kapa_ln_r_IH[i]
            taur_r_IH[i+2] = taur_r_IH[i]
            ###################################################################
            #                        Other parameters                         #
            ###################################################################
            tallian_mat_fac_r[i+2] = tallian_mat_fac_r[i]
            shear_limt_r[i+2] = shear_limt_r[i]
            asp_ht_b_r[i+2] = asp_ht_b_r[i]
            asp_trac_b_r[i+2] = asp_trac_b_r[i]
            asp_slope_b_r[i+2] = asp_slope_b_r[i]
            sig_theta_r[i+2] = sig_theta_r[i]
            cont_fac_r[i+2] = cont_fac_r[i]
            proc_fac_r[i+2] = proc_fac_r[i]
            prod_fac_r[i+2] = prod_fac_r[i]
            taur_r_GZ[i+2] = taur_r_GZ[i]
            taur_r_IH[i+2] = taur_r_IH[i]
    ###########################################################################
    #             Other initialization constant for fatigue life              #
    ###########################################################################
    fl_beta, fl_zeta, fl_zeta_inv = 1.6, 7.33, 1 / 7.33
    phi_0_b, phi_1a, phi_1b, phi_1f = 0., 1.5, 1., 1.5
    phi_1b_LP, phi_2b_LP, phi_3b_LP = 1., 1., 1.6e-5
    b_zeta, b_zeta_inv = 11.728, 1 / 11.728
    dd, scf = D_b / D_m, 1.
    q_fac_LP = 1.
    """
    Uncomment the lines below for Jones stress
    concentration factor for roller bearings

    scf = 0.61
    if abs(Re_len - Re_cen_len) <= 1e-6):
        scf = 0.45
    """
    if k_life_base_type[0] > 0:
        con_ang_r = np.zeros(2)
        con_ang_r[0] = 0 # + taper_ang_o
        con_ang_r[1] = math.pi # + taper_ang_i

    E_b_r = np.zeros(max_rs)
    E_b_r[0] = (1 - po_b ** 2) / E_b + (1 - po_o ** 2) / E_o
    E_b_r[1] = (1 - po_b ** 2) / E_b + (1 - po_i ** 2) / E_i

    f_r = np.zeros(max_rs)
    f_r[0] = f_o
    f_r[1] = f_i

    E_M, phi_0, phi_2b, phi_2f, ds, qcc = (
        np.zeros(max_rs), np.zeros(max_rs), np.zeros(max_rs),
        np.zeros(max_rs), np.zeros(max_rs), np.zeros(max_rs)
    )
    for i in range(2):
        E_M[i] = 1 / E_b_r[i]
        phi_2b[i] = 1 * proc_fac_r[i]
        phi_2f[i] = 600 * cont_fac_r[i]
        ds[i] = 0.0306 * 40 / sig_theta_r[i] ** 2
        phi_0[i] = (shear_limt_r[i] * (3 ** b_zeta_inv) *
                    (10 ** (-45 * b_zeta_inv - 6)))
        #######################################################################
        #                       Point contact constants                       #
        #######################################################################
        if k_life_base_type[0] <= 0:
            if f_r[i] == 0:
                xx = 1.
            else:
                xx = 2 * f_r[i] / (2 * f_r[i] - 1)
            qcc[i] = (f_con_r[i] * 0.5 ** (1 / q_exp_r[i]) *
                      (xx ** 0.41) * (dd ** 0.3) * D_b ** 1.8)
        #######################################################################
        #                       Line contact constants                        #
        #######################################################################
        else:
            q_fac_LP = 1
            """
            uncomment the lines below for Jones factor for tapered roller
            bearings

            x_taper = 1 - 0.15 * tan_taper_r[i] * cos_taper_r[i]
            q_fac_LP = scf * x_taper
            """
            x_gama = dd * con_ang_r[i]
            qcc[i] = (fac_r_ori[i] * f_con_r[i] * 0.5 ** (1 / q_exp_r[i]) *
                      q_fac_LP * (dd ** 0.22222222) * D_b ** 1.074 *
                      (1 + x_gama) ** 1.0740741)

    phi_2b_b = 1 * tallian_proc_fac_b
    phi_2f_b = 600 * tallian_cont_fac_b

    sk = np.zeros(n)
    for i in range(0, n):
        sk[i] = (2 * math.pi * (i + 1) / n) ** (fl_beta - 1)

    # fla = 0.23
    coff_am = 12 * 0.23
    fl_beta_inv = 1 / fl_beta

    am_r = np.zeros(max_rs)
    am_r[0] = coff_am / tallian_mat_fac_r[0] ** fl_beta_inv
    am_r[1] = coff_am / tallian_mat_fac_r[1] ** fl_beta_inv

    am_b = coff_am / tallian_mat_fac_b ** fl_beta_inv

    stle_mat_fac_r = np.zeros(max_rs)
    stle_mat_fac_r[0] = stle_mat_fac_o
    stle_mat_fac_r[1] = stle_mat_fac_i

    stle_proc_fac_r = np.zeros(max_rs)
    stle_proc_fac_r[0] = stle_proc_fac_o
    stle_proc_fac_r[1] = stle_proc_fac_i

    stle_hard_fac_r = np.zeros(max_rs)
    stle_hard_fac_r[0] = stle_hard_fac_o
    stle_hard_fac_r[1] = stle_hard_fac_i

    tallian_mat_fac_r = np.zeros(max_rs)
    tallian_mat_fac_r[0] = tallian_mat_fac_o
    tallian_mat_fac_r[1] = tallian_mat_fac_i

    stle_mat_fac_r, stle_proc_fac_r, stle_hard_fac_r, tallian_mat_fac_r = (
        np.zeros(max_rs), np.zeros(max_rs), np.zeros(max_rs), np.zeros(max_rs)
    )
    ###########################################################################
    #               Set parameters for secondary race surfaces                #
    ###########################################################################
    if max_rs > 2:
        for i in range(0, 2):
            j = i + 2
            qcc[j] = qcc[i]
            am_r[j] = am_r[i]
            ###################################################################
            #                   Tallian Life Mod Parameters                   #
            ###################################################################
            ds[j] = ds[i]
            E_M[j] = E_M[i]
            phi_0[j] = phi_0[i]
            phi_2b[j] = phi_2b[i]
            phi_2f[j] = phi_2f[i]
            ###################################################################
            #                          STLE factors                           #
            ###################################################################
            stle_mat_fac_r[j] =  stle_mat_fac_r[i]
            stle_proc_fac_r[j] = stle_proc_fac_r[i]
            stle_hard_fac_r[j] = stle_hard_fac_r[i]
            tallian_mat_fac_r[j] =  tallian_mat_fac_r[i]
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    Info_flc = (E_base,
                # Base line elastic modulus (52100 at room temp).
                E_M,
                # Base elastic constant.
                E_P_base,
                # Elastic modulus parameter corresponding to base.
                E_R,
                # Elastic constant ratios for output.
                am_b,
                # Function of constants A and M for rolling elements.
                am_r,
                # Function of constants A and M for races.
                asp_ht_b_r,
                # Rms asperity height (mm), composite roughness of race.
                asp_trac_b_r,
                # Asperity traction coefficient.
                asp_slope_b_r,
                # Asperity slope.
                b_zeta_inv,
                # 1/b_zeta.
                c_kapa_ln_b_IH,
                # Rolling element multiplier for IH constant for line contact.
                c_kapa_pt_b_IH,
                # Rolling element multiplier for IH constant for point contact.
                c_kapa_ln_b_GZ,
                # Rolling element multiplier for GZ constant for line contact.
                c_kapa_pt_b_GZ,
                # Rolling element multiplier for GZ constant for point contact.
                c_kapa_ln_b_LP,
                # Rolling element multiplier for LP constant for line contact.
                c_kapa_pt_b_LP,
                # Rolling element multiplier for LP constant for point contact.
                c_kapa_ln_r_IH,
                # Race multiplier for IH constant for line contact.
                c_kapa_pt_r_IH,
                # Race multiplier for IH constant for point contact.
                c_kapa_ln_r_GZ,
                # Race multiplier for GZ constant for line contact.
                c_kapa_pt_r_GZ,
                # Race multiplier for GZ constant for point contact.
                c_kapa_ln_r_LP,
                # Race multiplier for LP constant for line contact.
                c_kapa_pt_r_LP,
                # Race multiplier for LP constant for point contact.
                dd,
                # Rolling element dia to pitch dia ratio.
                depth_exp_b_IH,
                # Current IH rolling element depth exponent.
                depth_exp_r_IH,
                # Current IH race depth exponent.
                ds,
                # Asperity count D_s.
                f_con_b_IH,
                # Default IH stress capacity rolling element factor.
                f_con_b_GZ,
                # Default GZ stress capacity rolling element factor.
                f_con_b_LP,
                # Default LP stress capacity rolling element factor.
                f_con_r_IH,
                # Operating IH fatigue constant of race.
                f_con_r_GZ,
                # Operating GZ fatigue constant of race.
                f_con_r_LP,
                # Operating LP fatigue constant of race.
                fl_beta,
                # Life dispersion exponent, beta.
                fl_zeta,
                # Stress/life exponent, zeta.
                fl_zeta_inv,
                # 1/fl_zeta.
                g_exp_b_IH,
                # Geometric parameter exponent in IH stress capacity equation
                # of rolling element.
                g_exp_b_GZ,
                # Geometric parameter exponent in GZ stress capacity equation
                # of rolling element.
                g_exp_b_LP,
                # Geometric parameter exponent in LP stress capacity equation
                # of rolling element.
                g_exp_r_IH,
                # Geometric parameter exponent in IH stress capacity equation
                # of race.
                g_exp_r_GZ,
                # Geometric parameter exponent in GZ stress capacity equation
                # of race.
                g_exp_r_LP,
                # Geometric parameter exponent in LP stress capacity equation
                # of race.
                g0_exp_b_IH,
                # Exponent 1 in IH geometric parameter equation of
                # rolling element.
                g0_exp_b_GZ,
                # Exponent 1 in GZ geometric parameter equation of
                # rolling element.
                g0_exp_b_LP,
                # Exponent 1 in LP geometric parameter equation of
                # rolling element.
                g1_exp_b_IH,
                # Exponent 2 in IH geometric parameter equation of
                # rolling element.
                g1_exp_b_LP,
                # Exponent 2 in IH geometric parameter equation of
                # rolling element.
                g2_exp_b_IH,
                # Exponent 3 in IH geometric parameter equation of
                # rolling element.
                g2_exp_b_GZ,
                # Exponent 3 in IH geometric parameter equation of
                # rolling element.
                g2_exp_b_LP,
                # Exponent 3 in IH geometric parameter equation of
                # rolling element.
                g0_exp_r_IH,
                # Exponent 1 in IH geometric parameter equation of race.
                g0_exp_r_GZ,
                # Exponent 1 in GZ geometric parameter equation of race.
                g0_exp_r_LP,
                # Exponent 1 in LP geometric parameter equation of race.
                g1_exp_r_IH,
                # Exponent 2 in IH geometric parameter equation of race.
                g1_exp_r_LP,
                # Exponent 2 in LP geometric parameter equation of race.
                g2_exp_r_IH,
                # Exponent 3 in IH geometric parameter equation of race.
                g2_exp_r_GZ,
                # Exponent 3 in GZ geometric parameter equation of race.
                g2_exp_r_LP,
                # Exponent 3 in LP geometric parameter equation of race.
                k_life_base_type,
                # Current model type (0-point contact, 1-line contact).
                k_life_mod_fac_b,
                # STLE life modification code for rolling element.
                k_life_mod_fac_r,
                # STLE life modification code for race.
                p_exp_b_IH,
                # IH stress exponent for rolling element.
                p_exp_b_GZ,
                # GZ stress exponent for rolling element.
                p_exp_b_LP,
                # LP stress exponent for rolling element.
                p_exp_r_IH,
                # IH stress exponent for race.
                p_exp_r_GZ,
                # GZ stress exponent for race.
                p_exp_r_LP,
                # LP stress exponent for race.
                phi_0,
                # Baseline matrix susceptibility term.
                phi_0_b,
                # Baseline matrix susceptibility term for rolling element.
                phi_1a,
                # Severity balancing factor for asperity defects.
                phi_1b,
                # Severity balancing factor for subsurface defects.
                phi_1b_LP,
                # Lundberg-Palmgen severity balancing factor for subsurface
                # defects.
                phi_1f,
                # Severity balancing factor for furrow defects.
                phi_2b,
                # Surface defect severity factor.
                phi_2b_b,
                # Rolling element material defect severity factor.
                phi_2b_LP,
                # Lundberg-Palmgren steel processing factor.
                phi_2f,
                # Furrow defect severity factor.
                phi_2f_b,
                # Rolling element furrow defect severity factor.
                phi_3b_LP,
                # Baseline subsurface stress field factor.
                pr_exp_b_IH,
                # Property ratio exponent in IH stress capacity equation of
                # rolling element.
                pr_exp_b_GZ,
                # Property ratio exponent in GZ stress capacity equation of
                # rolling element.
                pr_exp_b_LP,
                # Property ratio exponent in LP stress capacity equation of
                # rolling element.
                pr_exp_r_IH,
                # Property ratio exponent in IH stress capacity equation of
                # race.
                pr_exp_r_GZ,
                # Property ratio exponent in GZ stress capacity equation of
                # race.
                pr_exp_r_LP,
                # Property ratio exponent in LP stress capacity equation of
                # race.
                proc_fac_b,
                # Processing multiplier for rolling element.
                proc_fac_r,
                # Processing multiplier for races.
                prod_fac_b,
                # Product of all factors for rolling element.
                prod_fac_r,
                # Product of all factors for races.
                q_exp_len_r,
                # Corresponding exponents for length in original LP model.
                q_exp_len_r_GZ,
                # Corresponding exponents for length in GZ model.
                q_exp_len_r_LP,
                # Corresponding exponents for length in updated LP model.
                q_exp_load_r,
                # Corresponding exponents for load in original LP model.
                q_exp_load_r_GZ,
                # Corresponding exponents for load in GZ model.
                q_exp_load_r_LP,
                # Corresponding exponents for load in updated LP model.
                q_exp_load_r_inv,
                # Inverse of corresponding exponents for load in original LP
                # model.
                q_exp_load_r_GZ_inv,
                # Inverse of corresponding exponents for load in GZ model.
                q_exp_load_r_LP_inv,
                # Inverse of corresponding exponents for load in updated LP
                # model.
                q_exp_r,
                # Operating original LP load exponent.
                q_exp_r_inv,
                # Operating original LP load exponent inverse.
                qcc,
                # Constant parts of original LP load capicity eqn.
                shear_exp_b_IH,
                # IH shear stress exponent for rolling element.
                shear_exp_r_IH,
                # IH shear stress exponent for rolling element.
                shear_exp_r_GZ,
                # GZ shear stress exponent for rolling element.
                sk,
                # Constants array used in life modifying computations.
                taur_r_IH,
                # max shear stress due to residual stress in races,
                # IH positive value for compressive residual stress.
                taur_r_GZ,
                # max shear stress due to residual stress in races,
                # GZ positive value for compressive residual stress.
                u_exp_b_IH,
                # Contact frequency exponent in IH stress capacity equation.
                u_exp_b_GZ,
                # Contact frequency exponent in GZ stress capacity equation.
                u_exp_b_LP,
                # Contact frequency exponent in LP stress capacity equation.
                u_exp_r_IH,
                # Contact frequency exponent in IH stress capacity equation.
                u_exp_r_GZ,
                # Contact frequency exponent in GZ stress capacity equation.
                u_exp_r_LP,
                # Contact frequency exponent in LP stress capacity equation.
                wb_dis_b_inv,
                # Weibull dispersion constant for rolling element inverse.
                wb_dis_brg,
                # Weibull slope for composite bearing.
                wb_dis_brg_inv,
                # Weibull slope for composite bearing inverse.
                wb_dis_r,
                # Weibull dispersion constant for races.
                wb_dis_r_inv,
                # Weibull dispersion constant for races inverse.
                wb_s,
                # All Weibull slopes (original L-P model).
                wb_s_inv,
                # All Weibull slopes inverse (original L-P model).
                zeta_IH,
                # Ratio of max shear stress to Hertz pressure in IH model.
                zeta_GZ,
                # Ratio of max shear stress to Hertz pressure in GZ model.
                )

    return Info_flc

###############################################################################
#                           Calculate fatigue_life                            #
###############################################################################
# @njit(fastmath=False)
def fatigue_life(x, Info_es, Info_brcs, Info_brcs_, Info_brcf, Info_brcf_,
                 Info_brtf, Info_brtf_, Info_flc, mod_fl):
    """Solve the fatigue life.

    Parameters
    ----------
    x: np.darray
        Solution vector.
    Info_es: tuple
        Information of expansion_subcall.
    Info_brcs: tuple
        Information of ball_race_contact_strain.
    Info_brcs_: tuple
        Information of ball_race_contact_strain_.
    Info_brcf: tuple
        Information data of ball_race_contact_force.
    Info_brcf_: tuple
        Information data of ball_race_contact_force_.
    Info_brtf: tuple
        Information of ball_race_traction_force.
    Info_brtf_: tuple
        Information of ball_race_traction_force_.
    Info_flc: tuple
        Information data of fatigue_life_constant.
    mod_fl: tuple
        Mode data of fatigue_life.

    Returns
    -------
    Info_flc: tuple
        Information data of fatigue_life_constant.
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    (D_b,
     D_m,
     R_b,
     a_b_i,
     a_b_o,
     b_b_i,
     b_b_o,
     f_i,
     f_o,
     max_rs,
     n,
     str_limt_fac,
     von_mises_stress_b,
     von_mises_stress_i,
     von_mises_stress_o
     ) = mod_fl[0::]
    ###########################################################################
    #                               End Prepare                               #
    ###########################################################################
    (E_base,
     E_M,
     E_P_base,
     E_R,
     am_b,
     am_r,
     asp_ht_b_r,
     asp_trac_b_r,
     asp_slope_b_r,
     b_zeta_inv,
     c_kapa_ln_b_IH,
     c_kapa_pt_b_IH,
     c_kapa_ln_b_GZ,
     c_kapa_pt_b_GZ,
     c_kapa_ln_b_LP,
     c_kapa_pt_b_LP,
     c_kapa_ln_r_IH,
     c_kapa_pt_r_IH,
     c_kapa_ln_r_GZ,
     c_kapa_pt_r_GZ,
     c_kapa_ln_r_LP,
     c_kapa_pt_r_LP,
     dd,
     depth_exp_b_IH,
     depth_exp_r_IH,
     ds,
     f_con_b_IH,
     f_con_b_GZ,
     f_con_b_LP,
     f_con_r_IH,
     f_con_r_GZ,
     f_con_r_LP,
     fl_beta,
     fl_zeta,
     fl_zeta_inv,
     g_exp_b_IH,
     g_exp_b_GZ,
     g_exp_b_LP,
     g_exp_r_IH,
     g_exp_r_GZ,
     g_exp_r_LP,
     g0_exp_b_IH,
     g0_exp_b_GZ,
     g0_exp_b_LP,
     g1_exp_b_IH,
     g1_exp_b_LP,
     g2_exp_b_IH,
     g2_exp_b_GZ,
     g2_exp_b_LP,
     g0_exp_r_IH,
     g0_exp_r_GZ,
     g0_exp_r_LP,
     g1_exp_r_IH,
     g1_exp_r_LP,
     g2_exp_r_IH,
     g2_exp_r_GZ,
     g2_exp_r_LP,
     k_life_base_type,
     k_life_mod_fac_b,
     k_life_mod_fac_r,
     p_exp_b_IH,
     p_exp_b_GZ,
     p_exp_b_LP,
     p_exp_r_IH,
     p_exp_r_GZ,
     p_exp_r_LP,
     phi_0,
     phi_0_b,
     phi_1a,
     phi_1b,
     phi_1b_LP,
     phi_1f,
     phi_2b,
     phi_2b_b,
     phi_2b_LP,
     phi_2f,
     phi_2f_b,
     phi_3b_LP,
     pr_exp_b_IH,
     pr_exp_b_GZ,
     pr_exp_b_LP,
     pr_exp_r_IH,
     pr_exp_r_GZ,
     pr_exp_r_LP,
     proc_fac_b,
     proc_fac_r,
     prod_fac_b,
     prod_fac_r,
     q_exp_len_r,
     q_exp_len_r_GZ,
     q_exp_len_r_LP,
     q_exp_load_r,
     q_exp_load_r_GZ,
     q_exp_load_r_LP,
     q_exp_load_r_inv,
     q_exp_load_r_GZ_inv,
     q_exp_load_r_LP_inv,
     q_exp_r,
     q_exp_r_inv,
     qcc,
     shear_exp_b_IH,
     shear_exp_r_IH,
     shear_exp_r_GZ,
     sk,
     taur_IH,
     taur_GZ,
     u_exp_b_IH,
     u_exp_b_GZ,
     u_exp_b_LP,
     u_exp_r_IH,
     u_exp_r_GZ,
     u_exp_r_LP,
     wb_dis_b_inv,
     wb_dis_brg,
     wb_dis_brg_inv,
     wb_dis_r,
     wb_dis_r_inv,
     wb_s,
     wb_s_inv,
     zeta_IH,
     zeta_GZ) =Info_flc[0::]

    tauh_o_GZ = Info_es[18]
    tauh_i_GZ = Info_es[19]
    tauh_o_IH = Info_es[20]
    tauh_i_IH = Info_es[21]

    alpha_b_o_0 = Info_brcs[20]
    alpha_b_i_0 = Info_brcs[22]

    aa_b_o = Info_brcf[2]
    aa_b_i = Info_brcf[3]
    p_b_o_max = Info_brcf[6]
    p_b_i_max = Info_brcf[7]
    # q_b_o = Info_brcf[8]
    # q_b_i = Info_brcf[9]
    Q_b_o = Info_brcf[10]
    Q_b_i = Info_brcf[11]

    omega_b_b = Info_brtf[2]
    omega_o_o = Info_brtf[3]
    omega_i_i = Info_brtf[4]
    h_ts_b_o = Info_brtf[15]
    h_ts_b_i = Info_brtf[16]
    r_os_og_o_ctr = Info_brtf[19]
    r_is_ig_i_ctr = Info_brtf[20]
    miu_o_oc = Info_brtf[21]
    miu_i_ic = Info_brtf[22]

    if max_rs == 4:
        alpha_b_o_0_ = Info_brcs_[4]
        alpha_b_i_0_ = Info_brcs_[6]

        aa_b_o_ = Info_brcf_[2]
        aa_b_i_ = Info_brcf_[3]
        p_b_o_max_ = Info_brcf_[6]
        p_b_i_max_ = Info_brcf_[7]
        q_b_o_ = Info_brcf_[8]
        q_b_i_ = Info_brcf_[9]
        Q_b_o_ = Info_brcf_[10]
        Q_b_i_ = Info_brcf_[11]

        h_ts_b_o_ = Info_brtf_[10]
        h_ts_b_i_ = Info_brtf_[11]
        r_os_og_o_ctr_ = Info_brtf_[14]
        r_is_ig_i_ctr_ = Info_brtf_[15]
        miu_o_oc_ = Info_brtf_[16]
        miu_i_ic_ = Info_brtf_[17]


    con_ang_b_r = np.zeros((n, max_rs))
    con_ang_b_r[:, 0] = alpha_b_o_0[:, 0, 0]
    con_ang_b_r[:, 1] = alpha_b_i_0[:, 0, 0]

    cos_con_ang_b_r_0 = np.cos(con_ang_b_r[:, 0])
    cos_con_ang_b_r_1 = np.cos(math.pi - con_ang_b_r[:, 1])

    term_0_0 = -1 / ((D_m / cos_con_ang_b_r_0 + D_b) * 0.5)
    term_1_0 = -1 / (f_o * D_b)

    term_0_1 = 1 / ((D_m / cos_con_ang_b_r_1 - D_b) * 0.5)
    term_1_1 = -1 / (f_i * D_b)

    term_2_0 = 2 / R_b
    term_2_1 = term_2_0

    sum_b_r = np.zeros((n, max_rs))
    sum_b_r[:, 0] = term_0_0 + term_1_0 + term_2_0
    sum_b_r[:, 1] = term_0_1 + term_1_1 + term_2_1

    a_star = np.zeros((n, max_rs))
    a_star[:, 0] = a_b_o
    a_star[:, 1] = a_b_i

    b_star = np.zeros((n, max_rs))
    b_star[:, 0] = b_b_o
    b_star[:, 1] = b_b_i

    aa_b_r = np.zeros((n, max_rs))
    aa_b_r[:, 0] = aa_b_o[:, 0, 0]
    aa_b_r[:, 1] = aa_b_i[:, 0, 0]

    Q_b_r = np.zeros((n, max_rs))
    Q_b_r[:, 0] = Q_b_o[:, 0, 0]
    Q_b_r[:, 1] = Q_b_i[:, 0, 0]

    p_b_r_max = np.zeros((n, max_rs))
    p_b_r_max[:, 0] = p_b_o_max[:, 0, 0]
    p_b_r_max[:, 1] = p_b_i_max[:, 0, 0]

    omega_b = np.zeros((3, n))
    omega_b[0, :] = omega_b_b[:, 0, 0]
    omega_b[1, :] = omega_b_b[:, 1, 0]
    omega_b[2, :] = omega_b_b[:, 2, 0]

    omega_r = np.zeros((3, max_rs))
    omega_r[:, 0] = omega_o_o[0, :, 0]
    omega_r[:, 1] = omega_i_i[0, :, 0]

    film_b_r = np.zeros((n, max_rs))
    film_b_r[:, 0] = h_ts_b_o[:, 0, 0]
    film_b_r[:, 1] = h_ts_b_i[:, 0, 0]

    rad_con_pos_b_r = np.zeros((n, max_rs))
    rad_con_pos_b_r[:, 0] = np.sqrt(r_os_og_o_ctr[:, 1, 0] ** 2 +
                                    r_os_og_o_ctr[:, 2, 0] ** 2)
    rad_con_pos_b_r[:, 1] = np.sqrt(r_is_ig_i_ctr[:, 1, 0] ** 2 +
                                    r_is_ig_i_ctr[:, 2, 0] ** 2)

    miu_b_r_max = np.zeros((n, max_rs))
    for j in range(n):
        miu_b_r_max[j, 0] = np.max((miu_o_oc[j, 0, :] ** 2 +
                                    miu_o_oc[j, 1, :] ** 2) ** 0.5)
        miu_b_r_max[j, 1] = np.max((miu_i_ic[j, 0, :] ** 2 +
                                    miu_i_ic[j, 1, :] ** 2) ** 0.5)

    shear_limt_b_IH = von_mises_stress_b * 2 ** 0.5 / 3 * str_limt_fac

    shear_limt_r_IH = np.zeros(max_rs)
    shear_limt_r_IH[0] = von_mises_stress_o * 2 ** 0.5 / 3 * str_limt_fac
    shear_limt_r_IH[1] = von_mises_stress_i * 2 ** 0.5 / 3 * str_limt_fac

    tauh_r_GZ = np.zeros(max_rs)
    tauh_r_GZ[0] = tauh_o_GZ
    tauh_r_GZ[1] = tauh_i_GZ

    tauh_r_IH = np.zeros(max_rs)
    tauh_r_IH[0] = tauh_o_IH
    tauh_r_IH[1] = tauh_i_IH

    k_rot_race = np.zeros(max_rs)
    for i in range(2):
        if omega_r[0, i] == 0:
            k_rot_race[i] = 0
        else:
            k_rot_race[i] = 1

    if max_rs == 4:
        con_ang_b_r[:, 2] = alpha_b_o_0_[:, 0, 0]
        con_ang_b_r[:, 3] = alpha_b_i_0_[:, 0, 0]

        cos_con_ang_b_r_2 = np.cos(con_ang_b_r[:, 2])
        cos_con_ang_b_r_3 = np.cos(math.pi - con_ang_b_r[:, 3])

        term_0_2 = -1 / ((D_m / cos_con_ang_b_r_2 + D_b) * 0.5)
        term_1_2 = term_1_0

        term_0_3 = 1 / ((D_m / cos_con_ang_b_r_3 - D_b) * 0.5)
        term_1_3 = term_1_1

        term_2_2 = 2 / R_b
        term_2_3 = term_2_2

        sum_b_r[:, 2] = term_0_2 + term_1_2 + term_2_2
        sum_b_r[:, 3] = term_0_3 + term_1_3 + term_2_3

        a_star[:, 2] = a_star[:, 0]
        a_star[:, 3] = a_star[:, 1]

        b_star[:, 2] = b_star[:, 0]
        b_star[:, 3] = b_star[:, 1]

        aa_b_r[:, 2] = aa_b_o_[:, 0, 0]
        aa_b_r[:, 3] = aa_b_i_[:, 0, 0]

        Q_b_r[:, 2] = Q_b_o_[:, 0, 0]
        Q_b_r[:, 3] = Q_b_i_[:, 0, 0]

        p_b_r_max[:, 2] = p_b_o_max_[:, 0, 0]
        p_b_r_max[:, 3] = p_b_i_max_[:, 0, 0]

        omega_r[:, 2] = omega_r[:, 0]
        omega_r[:, 3] = omega_r[:, 1]

        film_b_r[:, 2] = h_ts_b_o_[:, 0, 0]
        film_b_r[:, 3] = h_ts_b_i_[:, 0, 0]

        rad_con_pos_b_r[:, 2] = np.sqrt(r_os_og_o_ctr_[:, 1, 0] ** 2 +
                                        r_os_og_o_ctr_[:, 2, 0] ** 2)
        rad_con_pos_b_r[:, 3] = np.sqrt(r_is_ig_i_ctr_[:, 1, 0] ** 2 +
                                        r_is_ig_i_ctr_[:, 2, 0] ** 2)

        for j in range(n):
            miu_b_r_max[j, 2] = np.max((miu_o_oc_[j, 0, :] ** 2+
                                        miu_o_oc_[j, 1, :] ** 2) ** 0.5)
            miu_b_r_max[j, 3] = np.max((miu_i_ic_[j, 0, :] ** 2+
                                        miu_i_ic_[j, 1, :] ** 2) ** 0.5)

        tauh_r_GZ[2] = tauh_o_GZ
        tauh_r_GZ[3] = tauh_i_GZ

        shear_limt_r_IH[2] = shear_limt_r_IH[0]
        shear_limt_r_IH[3] = shear_limt_r_IH[1]

        k_rot_race[2] = k_rot_race[0]
        k_rot_race[3] = k_rot_race[1]

    k_con_laod = np.zeros(2)
    if np.all(Q_b_r[:, 0:4:2] <= 0) == True:
        k_con_laod[0] = 1
    if np.all(Q_b_r[:, 1:4:2] <= 0) == True:
        k_con_laod[1] = 1

    (
        life_brg, life_r,
        life_brg_LP, life_r_LP, life_b_LP,
        life_brg_GZ, life_r_GZ, life_b_GZ,
        life_brg_IH, life_r_IH, life_b_IH
        ) = (
            np.zeros((1, 1, 3)), np.zeros((2, 1, 3)),
            np.zeros((1, 1, 3)), np.zeros((2, 1, 3)), np.zeros((n, 1, 3)),
            np.zeros((1, 1, 3)), np.zeros((2, 1, 3)), np.zeros((n, 1, 3)),
            np.zeros((1, 1, 3)), np.zeros((2, 1, 3)), np.zeros((n, 1, 3))
            )
    ###########################################################################
    #                       Setup for life computation                        #
    ###########################################################################
    if np.all(k_con_laod) > 0:
        #######################################################################
        #                            Bearing life                             #
        #######################################################################
        life_brg[0, 0, :] = 1e20
        life_brg_LP[0, 0, :] = 1e20
        life_brg_GZ[0, 0, :] = 1e20
        life_brg_IH[0, 0, :] = 1e20
        #######################################################################
        #                              Race life                              #
        #######################################################################
        life_r[:, 0, :] = 1e20
        life_r_LP[:, 0, :] = 1e20
        life_r_GZ[:, 0, :] = 1e20
        life_r_IH[:, 0, :] = 1e20
        #######################################################################
        #                        Rolling element life                         #
        #######################################################################
        life_b_LP[:, 0, :] = 1e20
        life_b_GZ[:, 0, :] = 1e20
        life_b_IH[:, 0, :] = 1e20
        #######################################################################
        #                            Store result                             #
        #######################################################################
        Info_fl = (life_brg,
                   # Original LP bearing life.
                   life_brg_LP,
                   # Updated LP bearing life.
                   life_brg_GZ,
                   # GZ bearing life.
                   life_brg_IH,
                   # IH bearing life.
                   life_r,
                   # Original LP race fatigue life.
                   life_r_LP,
                   # Updated LP race fatigue life.
                   life_r_GZ,
                   # GZ race life.
                   life_r_IH,
                   # IH race life.
                   life_b_LP,
                   # Updated LP rolling element life.
                   life_b_GZ,
                   # GZ rolling element life.
                   life_b_IH,
                   # IH rolling element life.
                   )
        return Info_fl
    ###########################################################################
    #                Rolling element with max inner race load                 #
    ###########################################################################
    kp = 0
    xk = Q_b_r[0, 1]
    for i in range(1, n):
        if Q_b_r[i, 1] > xk:
            kp = i

    """
    kq = 2
    if Q_b_r[kp, 0] > Q_b_r[kp, 1]:
        kq = 1
    """
    ###########################################################################
    #               Set rotating race speed (larger to the two)               #
    ###########################################################################
    omga = 0.
    for i in range(2):
        rav = abs(omega_r[0, i])
        if rav > omga:
            omga = rav
    ###########################################################################
    #                     Basic Contact Life Computation                      #
    ###########################################################################
    ###########################################################################
    #              No life computations if races are stationary               #
    ###########################################################################
    if omga == 0:
        #######################################################################
        #                            Bearing life                             #
        #######################################################################
        life_brg[0, 0, :] = 1e20
        life_brg_LP[0, 0, :] = 1e20
        life_brg_GZ[0, 0, :] = 1e20
        life_brg_IH[0, 0, :] = 1e20
        #######################################################################
        #                              Race life                              #
        #######################################################################
        life_r[:, 0, :] = 1e20
        life_r_LP[:, 0, :] = 1e20
        life_r_GZ[:, 0, :] = 1e20
        life_r_IH[:, 0, :] = 1e20
        #######################################################################
        #                        Rolling element life                         #
        #######################################################################
        life_b_LP[:, 0, :] = 1e20
        life_b_GZ[:, 0, :] = 1e20
        life_b_IH[:, 0, :] = 1e20
        #######################################################################
        #                            Store result                             #
        #######################################################################
        Info_fl = (life_brg,
                   # Original LP bearing life.
                   life_brg_LP,
                   # Updated LP bearing life.
                   life_brg_GZ,
                   # GZ bearing life.
                   life_brg_IH,
                   # IH bearing life.
                   life_r,
                   # Original LP race fatigue life.
                   life_r_LP,
                   # Updated LP race fatigue life.
                   life_r_GZ,
                   # GZ race life.
                   life_r_IH,
                   # IH race life.
                   life_b_LP,
                   # Updated LP rolling element life.
                   life_b_GZ,
                   # GZ rolling element life.
                   life_b_IH,
                   # IH rolling element life.
                   )
        return Info_fl
    ###########################################################################
    #                 Life computations if races are rotating                 #
    ###########################################################################
    else:
        flc = 1 / (60 * omga * 30 / math.pi)
    ###########################################################################
    #                     Initialize rolling element life                     #
    ###########################################################################
    (
        life_b_LP_inv, life_b_mod_LP_inv, life_b_stle_LP_inv,
        life_b_GZ_inv, life_b_mod_GZ_inv, life_b_stle_GZ_inv,
        life_b_IH_inv, life_b_mod_IH_inv, life_b_stle_IH_inv
        ) = (
            np.zeros(n), np.zeros(n), np.zeros(n),
            np.zeros(n), np.zeros(n), np.zeros(n),
            np.zeros(n), np.zeros(n), np.zeros(n)
            )
    ###########################################################################
    #                          Initialize race life                           #
    ###########################################################################
    (
        life_r_inv, life_r_mod_inv, life_r_stle_inv,
        life_r_LP_inv, life_r_mod_LP_inv, life_r_stle_LP_inv,
        life_r_GZ_inv, life_r_mod_GZ_inv, life_r_stle_GZ_inv,
        life_r_IH_inv, life_r_mod_IH_inv, life_r_stle_IH_inv,
        ) = (
            np.zeros(max_rs), np.zeros(max_rs), np.zeros(max_rs),
            np.zeros(max_rs), np.zeros(max_rs), np.zeros(max_rs),
            np.zeros(max_rs), np.zeros(max_rs), np.zeros(max_rs),
            np.zeros(max_rs), np.zeros(max_rs), np.zeros(max_rs)
            )
    ###########################################################################
    #                         Start race segment loop                         #
    ###########################################################################
    (
        g_param_r_LP,
        p_cap_r_LP,
        q_cap_r_LP,
        g_param_b_LP,
        p_cap_b_LP,
        q_cap_b_LP,
        g_param_r_GZ,
        p_cap_r_GZ,
        q_cap_r_GZ,
        g_param_b_GZ,
        p_cap_b_GZ,
        q_cap_b_GZ,
        g_param_r_IH,
        p_cap_r_IH,
        q_cap_r_IH,
        g_param_b_IH,
        p_cap_b_IH,
        q_cap_b_IH,
        q_cap_r_max,
        p_cap_r_LP_max,
        q_cap_r_LP_max,
        p_cap_r_GZ_max,
        q_cap_r_GZ_max,
        p_cap_r_IH_max,
        q_cap_r_IH_max,
        p_cap_b_LP_max,
        q_cap_b_LP_max,
        p_cap_b_GZ_max,
        q_cap_b_GZ_max,
        p_cap_b_IH_max,
        q_cap_b_IH_max,
        stle_lamba,
        stle_lub_fac,
        phi_4,
        phi_4_LP,
        phi_4_GZ,
        phi_4_IH,
        phi_T,
        phi_T_LP,
        phi_T_GZ,
        phi_T_IH,
        phi_4_b_LP,
        phi_4_b_GZ,
        phi_4_b_IH,
        phi_T_b_LP,
        phi_T_b_GZ,
        phi_T_b_IH,
        life_bs_con_LP,
        life_bs_con_GZ,
        life_bs_con_IH,
        life_rs_con_LP,
        life_rs_con_GZ,
        life_rs_con_IH
        ) = (
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros(max_rs),
            np.zeros(max_rs),
            np.zeros(max_rs),
            np.zeros(max_rs),
            np.zeros(max_rs),
            np.zeros(max_rs),
            np.zeros(max_rs),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs)),
            np.zeros((n, max_rs))
            )
    for i in range(max_rs):
        #######################################################################
        #                       Start race segment loop                       #
        #######################################################################
        life_r_inv[i] = 0.
        life_r_LP_inv[i] = 0.
        life_r_GZ_inv[i] = 0.
        life_r_IH_inv[i] = 0.
        life_r_stle_inv[i] = 0.
        life_r_stle_LP_inv[i] = 0.
        life_r_stle_GZ_inv[i] = 0.
        life_r_stle_IH_inv[i] = 0.
        #######################################################################
        #                 Start rolling element segment loop                  #
        #######################################################################
        for j in range(n):
            u = abs(x[29+12*j] - omega_r[0, i]) / omga

            omega_b_norm = math.sqrt(omega_b[0, j] ** 2 +
                                     omega_b[1, j] ** 2 +
                                     omega_b[2, j] ** 2)
            u_b = abs(x[29+12*j] - omega_b_norm) / omga

            phi_4[j] = 0.
            phi_T[j] = 0.

            (
                life_b_con_LP, life_b_con_GZ, life_b_con_IH,
                life_r_con_LP,  life_r_con_GZ, life_r_con_IH
                ) = (
                    0., 0., 0.,
                    0., 0., 0.
                    )
            ###################################################################
            #                        Original LP model                        #
            ###################################################################
            if Q_b_r[j, i] > 0:
                ###############################################################
                #                        Point contact                        #
                ###############################################################
                if k_life_base_type[0] <= 0:
                    ###########################################################
                    #           Jones spin factor for ball bearings           #
                    ###########################################################
                    spin_fac = 1.
                    ###########################################################
                    #                Contact load and capacity                #
                    ###########################################################
                    q_e = Q_b_r[j, i]
                    q_cap = (
                        qcc[i] *
                        (1 + dd * math.cos(con_ang_b_r[j, i])) ** 1.39 *
                        spin_fac / u ** q_exp_r_inv[i]
                    )
                ###############################################################
                #                        Line contact                         #
                ###############################################################
                else:
                    pass
                    """
                    rot = int(k_rot_race[i])
                    q_life_inv = q_life[j, i] ** q_exp_load_r_inv[i, rot]
                    q_e = a_life[j, i] * q_life_inv
                    q_cap = (qcc[i] * (2 * aa_b_r[j, i]) ** 0.778 /
                             u ** q_exp_r_inv[i])
                    """
                ###############################################################
                #                Generalized LP and GZ models                 #
                ###############################################################
                if k_life_base_type[1] <= 0:
                    ###########################################################
                    #                      Point contact                      #
                    ###########################################################
                    ###########################################################
                    #                 LP race stress capacity                 #
                    ###########################################################
                    g_param_r_LP[j, i] = (2 * rad_con_pos_b_r[j,i] *
                                          a_star[j, i] ** g0_exp_r_LP[i] *
                                          b_star[j, i] ** g1_exp_r_LP[i] *
                                          sum_b_r[j, i] ** g2_exp_r_LP[i])
                    p_cap_r_LP[j, i] = (f_con_r_LP[i] *
                                        E_R[i] ** pr_exp_r_LP[i] *
                                        g_param_r_LP[j, i] ** g_exp_r_LP[i] *
                                        c_kapa_pt_r_LP[i] * u ** u_exp_r_LP[i])
                    q_cap_r_LP[j, i] = (
                        2 * (math.pi * p_cap_r_LP[j, i] *
                             a_star[j, i] * b_star[j, i]) ** 3 /
                        (3 * (sum_b_r[j, i] * E_P_base * E_R[i]) ** 2)
                    )
                    ###########################################################
                    #           LP rolling element stress capacity            #
                    ###########################################################
                    g_param_b_LP[j, i] = (D_b *
                                          a_star[j, i] ** g0_exp_b_LP *
                                          b_star[j, i] ** g1_exp_b_LP *
                                          sum_b_r[j, i] ** g2_exp_b_LP)
                    p_cap_b_LP[j, i] = (f_con_b_LP * E_R[i] ** pr_exp_b_LP *
                                        g_param_b_LP[j, i] ** g_exp_b_LP *
                                        c_kapa_pt_b_LP * u_b ** u_exp_b_LP)
                    q_cap_b_LP[j, i] = (
                        2 * (math.pi * p_cap_b_LP[j, i] *
                             a_star[j, i] * b_star[j, i]) ** 3 /
                        (3 * (sum_b_r[j, i] * E_P_base * E_R[i]) ** 2)
                    )
                    ###########################################################
                    #                 GZ race stress capacity                 #
                    ###########################################################
                    g_param_r_GZ[j, i] = (
                        2 * rad_con_pos_b_r[j, i] *
                        (a_star[j, i] * b_star[j, i]) ** g0_exp_r_GZ[i] *
                        sum_b_r[j, i] ** g2_exp_r_GZ[i]
                    )
                    p_cap_r_GZ[j, i] = (f_con_r_GZ[i] *
                                        E_R[i] ** pr_exp_r_GZ[i] *
                                        g_param_r_GZ[j, i] ** g_exp_r_GZ[i] *
                                        c_kapa_pt_r_GZ[i] * u ** u_exp_r_GZ[i])
                    q_cap_r_GZ[j, i] = (
                        q_cap_r_LP[j, i] *
                        (p_cap_r_GZ[j, i] / p_cap_r_LP[j, i]) ** 3
                    )
                    ###########################################################
                    #           GZ rolling element stress capacity            #
                    ###########################################################
                    g_param_b_GZ[j,i] = (
                        D_b *
                        (a_star[j, i] * b_star[j, i]) ** g0_exp_b_GZ *
                        sum_b_r[j, i] ** g2_exp_b_GZ
                    )
                    p_cap_b_GZ[j, i] = (f_con_b_GZ *
                                        E_R[i] ** pr_exp_b_GZ *
                                        g_param_b_GZ[j, i] ** g_exp_b_GZ *
                                        c_kapa_pt_b_GZ * u_b ** u_exp_b_GZ)
                    q_cap_b_GZ[j, i] = (
                        q_cap_b_LP[j, i] *
                        (p_cap_b_GZ[j, i] / p_cap_b_LP[j, i]) ** 3
                    )
                    ###########################################################
                    #                 IH race stress capacity                 #
                    ###########################################################
                    g_param_r_IH[j, i] = (2 * rad_con_pos_b_r[j, i] *
                                          a_star[j, i] ** g0_exp_r_IH[i] *
                                          b_star[j, i] ** g1_exp_r_IH[i] *
                                          sum_b_r[j, i] ** g2_exp_r_IH[i])
                    p_cap_r_IH[j, i] = (f_con_r_IH[i] *
                                        E_R[i] ** pr_exp_r_IH[i] *
                                        g_param_r_IH[j, i] ** g_exp_r_IH[i] *
                                        c_kapa_pt_r_IH[i] * u ** u_exp_r_IH[i])
                    q_cap_r_IH[j,i] = (
                        2 * (math.pi * p_cap_r_IH[j, i] *
                             a_star[j, i] * b_star[j, i]) ** 3 /
                        (3 * (sum_b_r[j, i] * E_P_base * E_R[i]) ** 2)
                    )
                    ###########################################################
                    #           IH rolling element stress capacity            #
                    ###########################################################
                    g_param_b_IH[j, i] = (D_b *
                                          a_star[j, i] ** g0_exp_b_IH *
                                          b_star[j,i] ** g1_exp_b_IH *
                                          sum_b_r[j, i] ** g2_exp_b_IH)
                    p_cap_b_IH[j,i] = (f_con_b_IH *
                                       E_R[i] ** pr_exp_b_IH *
                                       g_param_b_IH[j, i] ** g_exp_b_IH *
                                       c_kapa_pt_b_IH * u_b ** u_exp_b_IH)
                    q_cap_b_IH[j,i] = (
                        2 * (math.pi * p_cap_b_IH[j, i] *
                             a_star[j, i] * b_star[j, i]) ** 3 /
                        (3 * (sum_b_r[j, i] * E_P_base * E_R[i]) ** 2)
                    )
                else:
                    pass
                    """
                    ###########################################################
                    #                      Line contact                       #
                    ###########################################################
                    ###########################################################
                    #                 LP race stress capacity                 #
                    ###########################################################
                    g_param_r_LP[j, i] = (
                        2 * rad_con_pos_b_r[j, i] * aa_b_r[j, i] *
                        sum_b_r[j, i] ** g2_exp_r_LP[i]
                    )
                    p_cap_r_LP[j, i] = (f_con_r_LP[i] *
                                        E_R[i] ** pr_exp_r_LP[i] *
                                        g_param_r_LP[j, i] ** g_exp_r_LP[i] *
                                        c_kapa_ln_r_LP[i] * u ** u_exp_r_LP[i])
                    q_cap_r_LP[j, i] = (
                        2 * math.pi * aa_b_r[j, i] *
                        p_cap_r_LP[j,i] ** 2 / (sum_b_r[j, i] * E_P_base)
                    )
                    ###########################################################
                    #           LP rolling element stress capacity            #
                    ###########################################################
                    g_param_b_LP[j, i] = (D_b * aa_b_r[j, i] *
                                          sum_b_r[j, i] ** g2_exp_b_LP)
                    p_cap_b_LP[j, i] = (f_con_b_LP *
                                        E_R[i] ** pr_exp_b_LP *
                                        g_param_b_LP[j, i] ** g_exp_b_LP *
                                        c_kapa_ln_b_LP * u_b ** u_exp_b_LP)
                    q_cap_b_LP[j, i] = (
                        2 * math.pi * aa_b_r[j, i] *
                        p_cap_b_LP[j, i] ** 2 / (sum_b_r[j, i] * E_P_base)
                    )
                    ###########################################################
                    #                 GZ race stress capacity                 #
                    ###########################################################
                    g_param_r_GZ[j, i] = (2 * rad_con_pos_b_r[j, i] *
                                          aa_b_r[j, i] / sum_b_r[j, i])
                    p_cap_r_GZ[j, i] = (f_con_r_GZ[i] *
                                        E_R[i] ** pr_exp_r_GZ[i] *
                                        g_param_r_GZ[j, i] ** g_exp_r_GZ[i] *
                                        c_kapa_ln_r_GZ[i] * u ** u_exp_r_GZ[i])
                    q_cap_r_GZ[j,i] = (
                        q_cap_r_LP[j, i] *
                        (p_cap_r_GZ[j, i] / p_cap_r_LP[j, i]) ** 2
                    )
                    ###########################################################
                    #           GZ rolling element stress capacity            #
                    ###########################################################
                    g_param_b_GZ[j, i] = D_b * aa_b_r[j, i] / sum_b_r[j, i]
                    p_cap_b_GZ[j, i] = (f_con_b_GZ *
                                        E_R[i] ** pr_exp_b_GZ *
                                        g_param_b_GZ[j, i] ** g_exp_b_GZ *
                                        c_kapa_ln_b_GZ * u_b ** u_exp_b_GZ)
                    q_cap_b_GZ[j, i] = (
                        q_cap_b_LP[j, i] *
                        (p_cap_b_GZ[j, i] / p_cap_b_LP[j, i]) ** 2
                    )
                    ###########################################################
                    #                 IH race stress capacity                 #
                    ###########################################################
                    g_param_r_IH[j, i] = (
                        2 * rad_con_pos_b_r[j, i] * aa_b_r[j,i] *
                        sum_b_r[j, i] ** g2_exp_r_LP[i]
                    )
                    p_cap_r_IH[j, i] = (f_con_r_IH[i] *
                                        E_R[i] ** pr_exp_r_IH[i] *
                                        g_param_r_IH[j, i] ** g_exp_r_IH[i] *
                                        c_kapa_ln_r_IH[i] * u ** u_exp_r_IH[i])
                    q_cap_r_IH[j, i] = (
                        2 * math.pi * aa_b_r[j, i] *
                        p_cap_r_IH[j, i] ** 2 / (sum_b_r[j, i] * E_P_base)
                    )
                    ###########################################################
                    #           IH rolling element stress capacity            #
                    ###########################################################
                    g_param_b_IH[j, i] = (D_b * aa_b_r[j, i] *
                                          sum_b_r[j, i] ** g2_exp_b_IH)
                    p_cap_b_IH[j, i] = (f_con_b_IH *
                                        E_R[i] ** pr_exp_b_IH *
                                        g_param_b_IH[j, i] ** g_exp_b_IH *
                                        c_kapa_ln_b_IH * u_b ** u_exp_b_IH)
                    q_cap_b_IH[j, i] = (
                        2 * math.pi * aa_b_r[j, i] *
                        p_cap_b_IH[j, i] ** 2 / (sum_b_r[j,i] * E_P_base)
                    )
                ###############################################################
                #                    Compute contact life                     #
                ###############################################################
                ###############################################################
                #                      Original LP life                       #
                ###############################################################
                life_r_con = (q_e / q_cap) ** q_exp_r[i]
                ###############################################################
                #                           LP life                           #
                ###############################################################
                ratio_r_LP = p_b_r_max[j, i] / p_cap_r_LP[j, i]
                life_r_con_LP = ratio_r_LP ** p_exp_r_LP[i]

                ratio_b_LP = p_b_r_max[j, i] / p_cap_b_LP[j, i]
                life_b_con_LP = ratio_b_LP ** p_exp_b_LP
                ###############################################################
                #                           GZ life                           #
                ###############################################################
                tau_a = (1 - (taur_GZ[i] + tauh_r_GZ[i]) /
                         (p_b_r_max[j, i] * zeta_GZ))
                if tau_a > 0:
                    ratio_r_GZ = p_b_r_max[j, i] / p_cap_r_GZ[j, i]
                    life_r_con_GZ = (ratio_r_GZ ** p_exp_r_GZ[i] *
                                     tau_a ** shear_exp_r_GZ[i])
                else:
                    life_r_con_GZ = 0.

                ratio_b_GZ = p_b_r_max[j, i] / p_cap_b_GZ[j, i]
                life_b_con_GZ = ratio_b_GZ ** p_exp_b_GZ
                ###############################################################
                #                           IH life                           #
                ###############################################################
                pva = 1 - shear_limt_r_IH[i] / (p_b_r_max[j,i] * zeta_IH)
                if pva > 0:
                    tau_a = (1 - (
                        shear_limt_r_IH[i] + taur_IH[i] + tauh_r_IH[i]
                        ) / (p_b_r_max[j, i] * zeta_IH))
                    if tau_a > 0:
                        ratio_r_IH = p_b_r_max[j, i] / p_cap_r_IH[j, i]
                        life_r_con_IH = (
                            ratio_r_IH ** p_exp_r_IH[i] *
                            tau_a ** (shear_exp_r_IH[i]*wb_dis_r_inv[i]) *
                            pva ** ((2 - depth_exp_r_IH[i]) * wb_dis_r_inv[i])
                        )
                    else:
                        life_r_con_IH = 0.
                else:
                    life_r_con_IH = 0.

                tau_a = 1 - shear_limt_b_IH / (p_b_r_max[j, i] * zeta_IH)
                if tau_a > 0:
                    pva = tau_a
                    ratio_b_IH = p_b_r_max[j, i] / p_cap_b_IH[j, i]
                    life_b_con_IH = (
                        ratio_b_IH ** p_exp_b_IH *
                        tau_a ** (shear_exp_b_IH * wb_dis_b_inv) *
                        pva ** ((2 - depth_exp_b_IH) * wb_dis_b_inv)
                    )
                else:
                    life_b_con_IH = 0.
                """
                ###############################################################
                #                    Compute contact life                     #
                ###############################################################
                ###############################################################
                #                      Original LP life                       #
                ###############################################################
                life_r_con = (q_e / q_cap) ** q_exp_r[i]
                ###############################################################
                #                           LP life                           #
                ###############################################################
                ratio_r_LP = p_b_r_max[j, i] / p_cap_r_LP[j, i]
                life_r_con_LP = ratio_r_LP ** p_exp_r_LP[i]

                ratio_b_LP = p_b_r_max[j, i] / p_cap_b_LP[j, i]
                life_b_con_LP = ratio_b_LP ** p_exp_b_LP
                ###############################################################
                #                           GZ life                           #
                ###############################################################
                tau_a = (1 - (taur_GZ[i] + tauh_r_GZ[i]) /
                         (p_b_r_max[j, i] * zeta_GZ))
                if tau_a > 0:
                    ratio_r_GZ = p_b_r_max[j, i] / p_cap_r_GZ[j, i]
                    life_r_con_GZ = (ratio_r_GZ ** p_exp_r_GZ[i] *
                                     tau_a ** shear_exp_r_GZ[i])
                else:
                    life_r_con_GZ = 0.

                ratio_b_GZ = p_b_r_max[j, i] / p_cap_b_GZ[j, i]
                life_b_con_GZ = ratio_b_GZ ** p_exp_b_GZ
                ###############################################################
                #                           IH life                           #
                ###############################################################
                pva = 1 - shear_limt_r_IH[i] / (p_b_r_max[j,i] * zeta_IH)
                if pva > 0:
                    tau_a = (1 - (
                        shear_limt_r_IH[i] + taur_IH[i] + tauh_r_IH[i]
                        ) / (p_b_r_max[j, i] * zeta_IH))
                    if tau_a > 0:
                        ratio_r_IH = p_b_r_max[j, i] / p_cap_r_IH[j, i]
                        life_r_con_IH = (
                            ratio_r_IH ** p_exp_r_IH[i] *
                            tau_a ** (shear_exp_r_IH[i]*wb_dis_r_inv[i]) *
                            pva ** ((2 - depth_exp_r_IH[i]) * wb_dis_r_inv[i])
                        )
                    else:
                        life_r_con_IH = 0.
                else:
                    life_r_con_IH = 0.

                tau_a = 1 - shear_limt_b_IH / (p_b_r_max[j, i] * zeta_IH)
                if tau_a > 0:
                    pva = tau_a
                    ratio_b_IH = p_b_r_max[j, i] / p_cap_b_IH[j, i]
                    life_b_con_IH = (
                        ratio_b_IH ** p_exp_b_IH *
                        tau_a ** (shear_exp_b_IH * wb_dis_b_inv) *
                        pva ** ((2 - depth_exp_b_IH) * wb_dis_b_inv)
                    )
                else:
                    life_b_con_IH = 0.
                ###############################################################
                #                   STLE life modification                    #
                ###############################################################
                ###############################################################
                #               STLE life modification factor                 #
                ###############################################################
                stle_fac_r_inv, stle_fac_b_inv, a_lub_fac, a_lamba = (
                    1., 1., 0., 0.
                )

                if k_life_mod_fac_r[i] > 0:
                    a_lamba, a_lub_fac  = stle_lubrication_factor(
                        film_b_r[j,i], asp_ht_b_r[i]
                    )
                    stle_fac_r_inv = 1 / (a_lub_fac * prod_fac_r[i])

                if k_life_mod_fac_b > 0:
                    if k_life_mod_fac_r[i] > 0:
                        a_lamba, a_lub_fac  = stle_lubrication_factor(
                            film_b_r[j,i], asp_ht_b_r[i]
                        )
                    stle_fac_b_inv = 1 / (a_lub_fac * prod_fac_b)
                ###############################################################
                #           STLE life modified race contact life              #
                ###############################################################
                life_r_con_mod = life_r_con * stle_fac_r_inv
                life_r_con_mod_LP = life_r_con_LP * stle_fac_r_inv
                life_r_con_mod_GZ = life_r_con_GZ * stle_fac_r_inv
                life_r_con_mod_IH = life_r_con_IH * stle_fac_r_inv
                ###############################################################
                #      STLE life modified rolling element contact life        #
                ###############################################################
                life_b_con_mod_LP = life_b_con_LP * stle_fac_b_inv
                life_b_con_mod_GZ = life_b_con_GZ * stle_fac_b_inv
                life_b_con_mod_IH = life_b_con_IH * stle_fac_b_inv
                ###############################################################
                #             Summation for rolling element lives             #
                ###############################################################
                life_b_LP_inv[j] = life_b_LP_inv[j] + life_b_con_LP
                life_b_GZ_inv[j] = life_b_GZ_inv[j] + life_b_con_GZ
                life_b_IH_inv[j] = life_b_IH_inv[j] + life_b_con_IH

                life_b_stle_LP_inv[j] = (life_b_stle_LP_inv[j] +
                                         life_b_con_mod_LP)
                life_b_stle_GZ_inv[j] = (life_b_stle_GZ_inv[j] +
                                         life_b_con_mod_GZ)
                life_b_stle_IH_inv[j] = (life_b_stle_IH_inv[j] +
                                         life_b_con_mod_IH)
                ###############################################################
                #       Summation of base and stle modified race lives        #
                ###############################################################
                if k_rot_race[i] == 1:
                    life_r_inv[i] = life_r_inv[i] + life_r_con
                    life_r_LP_inv[i] = life_r_LP_inv[i] + life_r_con_LP
                    life_r_GZ_inv[i] = life_r_GZ_inv[i] + life_r_con_GZ
                    life_r_IH_inv[i] = life_r_IH_inv[i] + life_r_con_IH

                    life_r_stle_inv[i] = (life_r_stle_inv[i] +
                                          life_r_con_mod)
                    life_r_stle_LP_inv[i] = (life_r_stle_LP_inv[i] +
                                             life_r_con_mod_LP)
                    life_r_stle_GZ_inv[i] = (life_r_stle_GZ_inv[i] +
                                             life_r_con_mod_GZ)
                    life_r_stle_IH_inv[i] = (life_r_stle_IH_inv[i] +
                                             life_r_con_mod_IH)
                else:
                    life_r_inv[i] = (life_r_inv[i] +
                                     life_r_con ** wb_s)
                    life_r_LP_inv[i] = (life_r_LP_inv[i] +
                                        life_r_con_LP ** wb_dis_r[i])
                    life_r_GZ_inv[i] = (life_r_GZ_inv[i] +
                                        life_r_con_GZ ** wb_dis_r[i])
                    life_r_IH_inv[i] = (life_r_IH_inv[i] +
                                        life_r_con_IH ** wb_dis_r[i])

                    life_r_stle_inv[i] = (life_r_stle_inv[i] +
                                          life_r_con_mod ** wb_s)
                    life_r_stle_LP_inv[i] = (life_r_stle_LP_inv[i] +
                                             life_r_con_mod_LP ** wb_dis_r[i])
                    life_r_stle_GZ_inv[i] = (life_r_stle_GZ_inv[i] +
                                             life_r_con_mod_GZ ** wb_dis_r[i])
                    life_r_stle_IH_inv[i] = (life_r_stle_IH_inv[i] +
                                             life_r_con_mod_IH ** wb_dis_r[i])
                ###############################################################
                #              Tallian contact life modification              #
                ###############################################################
                ###############################################################
                #              Tallian life modification factor               #
                ###############################################################
                phi_2a, phi_3a, phi_3b, phi_3f = tallian_modification_factor(
                    film_b_r[j,i], asp_ht_b_r[i], ds[i], asp_trac_b_r[i],
                    asp_slope_b_r[i], E_M[i], p_b_r_max[j,i],
                    miu_b_r_max[j,i]
                )
                ###############################################################
                #                 Load distribution factors                   #
                ###############################################################
                phi_4[j] = life_r_con ** fl_beta
                phi_4_LP[j] = life_r_con_LP ** fl_beta
                phi_4_GZ[j] = life_r_con_GZ ** fl_beta
                phi_4_IH[j] = life_r_con_IH ** fl_beta
                ###############################################################
                #                     Race factors, phit                      #
                ###############################################################
                if k_life_mod_fac_r[i] > 0:
                    xx = (phi_1a * phi_2a * phi_3a +
                          phi_1b * phi_2b[i] * phi_3b +
                          phi_1f * phi_2f[i] * phi_3f)
                    phi_T[j] = xx * phi_4[j]
                    phi_T_LP[j] = xx * phi_4_LP[j]
                    phi_T_GZ[j] = xx * phi_4_GZ[j]
                    phi_T_IH[j] = xx * phi_4_IH[j]
                ###############################################################
                #                Rolling element factors, phit                #
                ###############################################################
                if k_life_mod_fac_b > 0:
                    xx = (phi_1a * phi_2a * phi_3a +
                          phi_1b * phi_2b_b * phi_3b +
                          phi_1f * phi_2f_b * phi_3f)
                    ###########################################################
                    #                        LP model                         #
                    ###########################################################
                    phi_4_b_LP[j,i] = phi_4_LP[j]
                    phi_T_b_LP[j,i] = xx * phi_4_LP[j]
                    ###########################################################
                    #                        GZ model                         #
                    ###########################################################
                    phi_4_b_GZ[j,i] = phi_4_GZ[j]
                    phi_T_b_GZ[j,i] = xx * phi_4_GZ[j]
                    ###########################################################
                    #                        IH model                         #
                    ###########################################################
                    phi_4_b_IH[j,i] = phi_4_IH[j]
                    phi_T_b_IH[j,i] = xx * phi_4_IH[j]
            ###################################################################
            #     Save parameters for heaviest loaded roll ele for output     #
            ###################################################################
            if j == kp:
                q_cap_r_max[i] = q_cap
                p_cap_r_LP_max[i] = p_cap_r_LP[j, i]
                q_cap_r_LP_max[i] = q_cap_r_LP[j, i]
                p_cap_r_GZ_max[i] = p_cap_r_GZ[j, i]
                q_cap_r_GZ_max[i] = q_cap_r_GZ[j, i]
                p_cap_r_IH_max[i] = p_cap_r_IH[j, i]
                q_cap_r_IH_max[i] = q_cap_r_IH[j, i]
                p_cap_b_LP_max[i] = p_cap_b_LP[j, i]
                q_cap_b_LP_max[i] = q_cap_b_LP[j, i]
                p_cap_b_GZ_max[i] = p_cap_b_GZ[j, i]
                q_cap_b_GZ_max[i] = q_cap_b_GZ[j, i]
                p_cap_b_IH_max[i] = p_cap_b_IH[j, i]
                q_cap_b_IH_max[i] = q_cap_b_IH[j, i]
                stle_lub_fac[i] = a_lub_fac
                stle_lamba[i] = a_lamba
            ###################################################################
            #           Save rolling element and race contact lives           #
            ###################################################################
            if life_b_con_LP > 0:
                life_bs_con_LP[j, i] = 1 / life_b_con_LP
                life_bs_con_GZ[j, i] = 1 / life_b_con_GZ
            else:
                life_bs_con_LP[j, i] = 1e20
                life_bs_con_GZ[j, i] = 1e20

            if life_r_con_LP > 0:
                life_rs_con_LP[j, i] = 1 / life_r_con_LP
            else:
                life_rs_con_LP[j, i] = 1e20

            if life_r_con_GZ > 0:
                life_rs_con_GZ[j, i] = 1 / life_r_con_GZ
            else:
                life_rs_con_GZ[j, i] = 1e20

            if life_b_con_IH > 0:
                life_bs_con_IH[j, i] = 1 / life_b_con_IH
            else:
                life_bs_con_IH[j, i] = 1e20

            if life_r_con_IH > 0:
                life_rs_con_IH[j, i] = 1 / life_r_con_IH
            else:
                life_rs_con_IH[j, i] = 1e20
        #######################################################################
        #                 Tallian life modification for races                 #
        #######################################################################
        if k_life_mod_fac_r[i] > 0:
            (
                phi_R, phi_4R,
                phi_R_LP, phi_4R_LP,
                phi_R_GZ, phi_4R_GZ,
                phi_R_IH, phi_4R_IH
                ) = (
                    0., 0.,
                    0., 0.,
                    0., 0.,
                    0., 0.
                    )
            ###################################################################
            #        Summation over rolling elements for race factors         #
            ###################################################################
            for k in range(n):
                if k_rot_race[i] == 0:
                    ###########################################################
                    #                    Original LP model                    #
                    ###########################################################
                    phi_R = phi_R + phi_T[k]
                    phi_4R = phi_4R + phi_4[k]
                    ###########################################################
                    #                  Generalized LP model                   #
                    ###########################################################
                    phi_R_LP = phi_R_LP + phi_T_LP[k]
                    phi_4R_LP = phi_4R_LP + phi_4_LP[k]
                    ###########################################################
                    #                        GZ model                         #
                    ###########################################################
                    phi_R_GZ = phi_R_GZ + phi_T_GZ[k]
                    phi_4R_GZ = phi_4R_GZ + phi_4_GZ[k]
                    ###########################################################
                    #                        IH model                         #
                    ###########################################################
                    phi_R_IH = phi_R_IH + phi_T_IH[k]
                    phi_4R_IH = phi_4R_IH + phi_4_IH[k]
                else:
                    ###########################################################
                    #                    Original LP model                    #
                    ###########################################################
                    phi_R = phi_R + phi_T[k] * sk[k]
                    phi_4R = phi_4R + phi_4[k] * sk[k]
                    ###########################################################
                    #                  Generalized LP model                   #
                    ###########################################################
                    phi_R_LP = phi_R_LP + phi_T_LP[k] * sk[k]
                    phi_4R_LP = phi_4R_LP + phi_4_LP[k] * sk[k]
                    ###########################################################
                    #                        GZ model                         #
                    ###########################################################
                    phi_R_GZ = phi_R_GZ + phi_T_GZ[k] * sk[k]
                    phi_4R_GZ = phi_4R_GZ + phi_4_GZ[k] * sk[k]
                    ###########################################################
                    #                        IH model                         #
                    ###########################################################
                    phi_R_IH = phi_R_IH + phi_T_IH[k] * sk[k]
                    phi_4R_IH = phi_4R_IH + phi_4_IH[k] * sk[k]
            ###################################################################
            #               Compute Tallian modified race life                #
            ###################################################################
            if phi_4R != 0:
                phi = phi_R / (phi_1b_LP * phi_2b_LP * phi_3b_LP * phi_4R)
                xx0 = life_r_inv[i] ** fl_zeta_inv
                xx = phi ** b_zeta_inv * xx0 - phi_0[i]
                life_r_mod_inv[i] = xx ** fl_zeta / am_r[i]
            else:
                life_r_mod_inv[i] = life_r_inv[i]

            if phi_4R_LP != 0:
                phi = phi_R_LP / (phi_2b_LP * phi_3b_LP * phi_4R_LP)
                xx0 = life_r_LP_inv[i] ** fl_zeta_inv
                xx = phi ** b_zeta_inv * xx0 - phi_0[i]
                life_r_mod_LP_inv[i] = xx ** fl_zeta / am_r[i]
            else:
                life_r_mod_LP_inv[i] = life_r_LP_inv[i]

            if phi_4R_GZ != 0:
                phi = phi_R_GZ / (phi_2b_LP * phi_3b_LP * phi_4R_GZ)
                xx0 = life_r_GZ_inv[i] ** fl_zeta_inv
                xx = phi ** b_zeta_inv * xx0 - phi_0[i]
                life_r_mod_GZ_inv[i] = xx ** fl_zeta / am_r[i]
            else:
                life_r_mod_GZ_inv[i] = life_r_GZ_inv[i]

            if phi_4R_IH != 0:
                phi = phi_R_IH / (phi_2b_LP * phi_3b_LP * phi_4R_IH)
                xx0 = life_r_IH_inv[i] ** fl_zeta_inv
                xx = phi ** b_zeta_inv * xx0 - phi_0[i]
                life_r_mod_IH_inv[i] = xx ** fl_zeta / am_r[i]
            else:
                life_r_mod_IH_inv[i] = life_r_IH_inv[i]
        else:
            life_r_mod_inv[i] = life_r_inv[i]
            life_r_mod_LP_inv[i] = life_r_LP_inv[i]
            life_r_mod_GZ_inv[i] = life_r_GZ_inv[i]
            life_r_mod_IH_inv[i] = life_r_IH_inv[i]
        #######################################################################
        #                  Life exponent for stationary race                  #
        #######################################################################
        if k_rot_race[i] != 1:
            life_r_inv[i] = life_r_inv[i] ** wb_s_inv
            life_r_mod_inv[i] = life_r_mod_inv[i] ** wb_s_inv
            life_r_stle_inv[i] = life_r_stle_inv[i] ** wb_s_inv

            life_r_LP_inv[i] = life_r_LP_inv[i] ** wb_dis_r_inv[i]
            life_r_mod_LP_inv[i] = life_r_mod_LP_inv[i] ** wb_dis_r_inv[i]
            life_r_stle_LP_inv[i] = life_r_stle_LP_inv[i] ** wb_dis_r_inv[i]

            life_r_GZ_inv[i] = life_r_GZ_inv[i] ** wb_dis_r_inv[i]
            life_r_mod_GZ_inv[i] = life_r_mod_GZ_inv[i] ** wb_dis_r_inv[i]
            life_r_stle_GZ_inv[i] = life_r_stle_GZ_inv[i] ** wb_dis_r_inv[i]

            life_r_IH_inv[i] = life_r_IH_inv[i] ** wb_dis_r_inv[i]
            life_r_mod_IH_inv[i] = life_r_mod_IH_inv[i] ** wb_dis_r_inv[i]
            life_r_stle_IH_inv[i] = life_r_stle_IH_inv[i] ** wb_dis_r_inv[i]
    ###########################################################################
    #              Tallian life modification for rolling element              #
    ###########################################################################
    for j in range(n):
        (
            phi_R, phi_4R,
            phi_R_LP, phi_4R_LP,
            phi_R_GZ, phi_4R_GZ,
            phi_R_IH, phi_4R_IH
            ) = (
                0., 0.,
                0., 0.,
                0., 0.,
                0., 0.
                )
        #######################################################################
        #          Summation over races for rolling elements factors          #
        #######################################################################
        if k_life_mod_fac_b > 0:
            for k in range(max_rs):
                ###############################################################
                #                    Generalized LP model                     #
                ###############################################################
                phi_R_LP = phi_R_LP + phi_T_b_LP[j, k]
                phi_4R_LP = phi_4R_LP + phi_4_b_LP[j, k]
                ###############################################################
                #                          GZ model                           #
                ###############################################################
                phi_R_GZ = phi_R_GZ + phi_T_b_GZ[j, k]
                phi_4R_GZ = phi_4R_GZ + phi_4_b_GZ[j, k]
                ###############################################################
                #                          IH model                           #
                ###############################################################
                phi_R_IH = phi_R_IH + phi_T_b_IH[j, k]
                phi_4R_IH = phi_4R_IH + phi_4_b_IH[j, k]
            ###################################################################
            #          Compute Tallian modified rolling element life          #
            ###################################################################
            ###################################################################
            #                      Generalized LP model                       #
            ###################################################################
            if phi_4R_LP != 0:
                phi = phi_R_LP / (phi_2b_LP * phi_3b_LP * phi_4R_LP)
                xx0 = life_b_LP_inv[j] ** fl_zeta_inv
                xx = phi ** b_zeta_inv * xx0 - phi_0_b
                life_b_mod_LP_inv[j] = xx ** fl_zeta / am_b
            else:
                life_b_mod_LP_inv[j] = life_b_LP_inv[j]
            ###################################################################
            #                            GZ model                             #
            ###################################################################
            if phi_4R_GZ != 0:
                phi = phi_R_GZ / (phi_2b_LP * phi_3b_LP * phi_4R_GZ)
                xx0 = life_b_GZ_inv[j] ** fl_zeta_inv
                xx = phi ** b_zeta_inv * xx0 - phi_0_b
                life_b_mod_GZ_inv[j] = xx ** fl_zeta / am_b
            else:
                life_b_mod_GZ_inv[j] = life_b_GZ_inv[j]
            ###################################################################
            #                            IH model                             #
            ###################################################################
            if phi_4R_IH != 0:
                phi = phi_R_IH / (phi_2b_LP * phi_3b_LP * phi_4R_IH)
                xx0 = life_b_IH_inv[j] ** fl_zeta_inv
                xx = phi ** b_zeta_inv * xx0 - phi_0_b
                life_b_mod_IH_inv[j] = xx ** fl_zeta / am_b
            else:
                life_b_mod_IH_inv[j] = life_b_IH_inv[j]
        else:
            life_b_mod_LP_inv[j] = life_b_LP_inv[j]
            life_b_mod_GZ_inv[j] = life_b_GZ_inv[j]
            life_b_mod_IH_inv[j] = life_b_IH_inv[j]
    ###########################################################################
    #                           Some house keeping                            #
    ###########################################################################
    ###########################################################################
    #       Save rolling element lub factor for heaviest loaded contact       #
    ###########################################################################
    # stle_lub_fac_b, stle_lamba_b = stle_lub_fac[kq], stle_lamba[kq]
    ###########################################################################
    #                       Compute total bearing life                        #
    ###########################################################################
    ###########################################################################
    #                     Summation over rolling elements                     #
    ###########################################################################
    f_ori_LP_inv = np.sum(life_b_LP_inv**wb_s)
    f_ori_GZ_inv = np.sum(life_b_GZ_inv**wb_s)
    f_ori_IH_inv = np.sum(life_b_IH_inv**wb_s)

    f_mod_LP_inv = np.sum(life_b_mod_LP_inv**wb_s)
    f_mod_GZ_inv = np.sum(life_b_mod_GZ_inv**wb_s)
    f_mod_IH_inv = np.sum(life_b_mod_IH_inv**wb_s)

    f_stle_LP_inv = np.sum(life_b_stle_LP_inv**wb_s)
    f_stle_GZ_inv = np.sum(life_b_stle_GZ_inv**wb_s)
    f_stle_IH_inv = np.sum(life_b_stle_IH_inv**wb_s)
    ###########################################################################
    #                      Start race loop for summation                      #
    ###########################################################################
    for i in range(2):
        if max_rs == 4:
            ###################################################################
            #                        Original LP model                        #
            ###################################################################
            life_r_inv[i] = life_r_inv[i] + life_r_inv[i+2]
            life_r_mod_inv[i] = life_r_mod_inv[i] + life_r_mod_inv[i+2]
            life_r_stle_inv[i] = life_r_stle_inv[i] + life_r_stle_inv[i+2]
            ###################################################################
            #                      Generalized LP model                       #
            ###################################################################
            life_r_LP_inv[i] = (life_r_LP_inv[i] +
                                life_r_LP_inv[i+2])
            life_r_mod_LP_inv[i] = (life_r_mod_LP_inv[i] +
                                    life_r_mod_LP_inv[i+2])
            life_r_stle_LP_inv[i] = (life_r_stle_LP_inv[i] +
                                     life_r_stle_LP_inv[i+2])
            ###################################################################
            #                            GZ model                             #
            ###################################################################
            life_r_GZ_inv[i] = (life_r_GZ_inv[i] +
                                life_r_GZ_inv[i+2])
            life_r_mod_GZ_inv[i] = (life_r_mod_GZ_inv[i] +
                                    life_r_mod_GZ_inv[i+2])
            life_r_stle_GZ_inv[i] = (life_r_stle_GZ_inv[i] +
                                     life_r_stle_GZ_inv[i+2])
            ###################################################################
            #                            IH model                             #
            ###################################################################
            life_r_IH_inv[i] = (life_r_IH_inv[i] +
                                life_r_IH_inv[i+2])
            life_r_mod_IH_inv[i] = (life_r_mod_IH_inv[i] +
                                    life_r_mod_IH_inv[i+2])
            life_r_stle_IH_inv[i] = (life_r_stle_IH_inv[i] +
                                     life_r_stle_IH_inv[i+2])
        #######################################################################
        #                     Invert and save race lives                      #
        #######################################################################
        if life_r_inv[i] > 0:
            life_r[i, 0, 0] = flc * 1e6 / life_r_inv[i]
            life_r[i, 0, 1] = flc * 1e6 / life_r_mod_inv[i]
            life_r[i, 0, 2] = flc * 1e6 / life_r_stle_inv[i]
            life_r_LP[i, 0, 0] = flc * 1e6 / life_r_LP_inv[i]
            life_r_LP[i, 0, 1] = flc * 1e6 / life_r_mod_LP_inv[i]
            life_r_LP[i, 0, 2] = flc * 1e6 / life_r_stle_LP_inv[i]
        else:
            life_r[i, 0, :] = 1e20
            life_r_LP[i, 0, :] = 1e20

        if life_r_GZ_inv[i] > 0:
            life_r_GZ[i, 0, 0] = flc * 1e6 / life_r_GZ_inv[i]
            life_r_GZ[i, 0, 1] = flc * 1e6 / life_r_mod_GZ_inv[i]
            life_r_GZ[i, 0, 2] = flc * 1e6 / life_r_stle_GZ_inv[i]
        else:
            life_r_GZ[i,0,:] = 1e20

        if life_r_IH_inv[i] > 0:
            life_r_IH[i, 0, 0] = flc * 1e6 / life_r_IH_inv[i]
            life_r_IH[i, 0, 1] = flc * 1e6 / life_r_mod_IH_inv[i]
            life_r_IH[i, 0, 2] = flc * 1e6 / life_r_stle_IH_inv[i]
        else:
            life_r_IH[i, 0, :] = 1e20
    ###########################################################################
    #              Summation over races for inverse bearing life              #
    ###########################################################################
    ###########################################################################
    #                            Original LP model                            #
    ###########################################################################
    f_ori_inv = np.sum(life_r_inv[0:2]**wb_s)
    f_mod_inv = np.sum(life_r_mod_inv[0:2]**wb_s)
    f_stle_inv = np.sum(life_r_stle_inv[0:2]**wb_s)
    ###########################################################################
    #                          Generalized LP model                           #
    ###########################################################################
    f_ori_LP_inv = (
        f_ori_LP_inv + np.sum(life_r_LP_inv[0:2]**wb_dis_brg)
    )
    f_mod_LP_inv = (
        f_mod_LP_inv + np.sum(life_r_mod_LP_inv[0:2]**wb_dis_brg)
    )
    f_stle_LP_inv = (
        f_stle_LP_inv + np.sum(life_r_stle_LP_inv[0:2]**wb_dis_brg)
    )
    ###########################################################################
    #                                GZ model                                 #
    ###########################################################################
    f_ori_GZ_inv = (
        f_ori_GZ_inv + np.sum(life_r_GZ_inv[0:2]**wb_dis_brg)
    )
    f_mod_GZ_inv = (
        f_mod_GZ_inv + np.sum(life_r_mod_GZ_inv[0:2]**wb_dis_brg)
    )
    f_stle_GZ_inv = (
        f_stle_GZ_inv + np.sum(life_r_stle_GZ_inv[0:2]**wb_dis_brg)
    )
    ###########################################################################
    #                                IH model                                 #
    ###########################################################################
    cond_0 = np.zeros(2)
    for i in range(2):
        if life_r_IH_inv[i] > 0:
            cond_0[i] = 1

    f_ori_IH_inv = (
        f_ori_IH_inv + np.sum(cond_0 * life_r_IH_inv[0:2]**wb_dis_brg)
    )

    cond_1 = np.zeros(2)
    for i in range(2):
        if life_r_mod_IH_inv[i] > 0:
            cond_1[i] = 1

    f_mod_IH_inv = (
        f_mod_IH_inv + np.sum(cond_1 * life_r_mod_IH_inv[0:2]**wb_dis_brg)
    )

    cond_2 = np.zeros(2)
    for i in range(2):
        if life_r_stle_IH_inv[i] > 0:
            cond_2[i] = 1

    f_stle_IH_inv = (
        f_stle_IH_inv + np.sum(cond_2 * life_r_stle_IH_inv[0:2]**wb_dis_brg)
    )
    ###########################################################################
    #                     Invert and save roll ele lives                      #
    ###########################################################################
    for j in range(n):
        if life_b_LP_inv[j] > 0:
            life_b_LP[j, 0, 0] = flc * 1e6 / life_b_LP_inv[j]
            life_b_LP[j, 0, 1] = flc * 1e6 / life_b_mod_LP_inv[j]
            life_b_LP[j, 0, 2] = flc * 1e6 / life_b_stle_LP_inv[j]
        else:
            life_b_LP[j, 0, :] = 1e20

        if life_b_GZ_inv[j] > 0:
            life_b_GZ[j, 0, 0] = flc * 1e6 / life_b_GZ_inv[j]
            life_b_GZ[j, 0, 1] = flc * 1e6 / life_b_mod_GZ_inv[j]
            life_b_GZ[j, 0, 2] = flc * 1e6 / life_b_stle_GZ_inv[j]
        else:
            life_b_GZ[j, 0, :] = 1e20

        if life_b_IH_inv[j] > 0:
            life_b_IH[j, 0, 0] = flc * 1e6 / life_b_IH_inv[j]
            life_b_IH[j, 0, 1] = flc * 1e6 / life_b_mod_IH_inv[j]
            life_b_IH[j, 0, 2] = flc * 1e6 / life_b_stle_IH_inv[j]
        else:
            life_b_IH[j, 0, :] = 1e20
        #######################################################################
        #                    Invert and save bearing life                     #
        #######################################################################
    if f_ori_inv > 0:
        #######################################################################
        #                          Original LP model                          #
        #######################################################################
        life_brg[0, 0, 0] = flc * 1e6 / f_ori_inv ** wb_s_inv
        life_brg[0, 0, 1] = flc * 1e6 / f_mod_inv ** wb_s_inv
        life_brg[0, 0, 2] = flc * 1e6 / f_stle_inv ** wb_s_inv
        #######################################################################
        #                        Generalized LP model                         #
        #######################################################################
        life_brg_LP[0, 0, 0] = flc * 1e6 / f_ori_LP_inv ** wb_dis_brg_inv
        life_brg_LP[0, 0, 1] = flc * 1e6 / f_mod_LP_inv ** wb_dis_brg_inv
        life_brg_LP[0, 0, 2] = flc * 1e6 / f_stle_LP_inv ** wb_dis_brg_inv
        #######################################################################
        #                              GZ model                               #
        #######################################################################
        life_brg_GZ[0, 0, 0] = flc * 1e6 / f_ori_GZ_inv ** wb_dis_brg_inv
        life_brg_GZ[0, 0, 1] = flc * 1e6 / f_mod_GZ_inv ** wb_dis_brg_inv
        life_brg_GZ[0, 0, 2] = flc * 1e6 / f_stle_GZ_inv ** wb_dis_brg_inv
    else:
        life_brg[0, 0, :], life_brg_LP[0, 0, :], life_brg_GZ[0, 0, :] = (
            1e20, 1e20, 1e20
        )
    ###########################################################################
    #                                IH model                                 #
    ###########################################################################
    if f_ori_IH_inv > 0:
        life_brg_IH[0, 0, 0] = flc * 1e6 / f_ori_IH_inv ** wb_dis_brg_inv
        life_brg_IH[0, 0, 1] = flc * 1e6 / f_mod_IH_inv ** wb_dis_brg_inv
        life_brg_IH[0, 0, 2] = flc * 1e6 / f_stle_IH_inv ** wb_dis_brg_inv
    else:
        life_brg_IH[0, 0, :] = 1e20
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    Info_fl = (life_brg,
               # Original LP bearing life.
               life_brg_LP,
               # Updated LP bearing life.
               life_brg_GZ,
               # GZ bearing life.
               life_brg_IH,
               # IH bearing life.
               life_r,
               # Original LP race fatigue life.
               life_r_LP,
               # Updated LP race fatigue life.
               life_r_GZ,
               # GZ race life.
               life_r_IH,
               # IH race life.
               life_b_LP,
               # Updated LP rolling element life.
               life_b_GZ,
               # GZ rolling element life.
               life_b_IH,
               # IH rolling element life.
               )
    return Info_fl

###############################################################################
#                                Dimensionless                                #
###############################################################################
# @njit(fastmath=False)
def dimensionless(a, b):
    """Solve the dimensionless array.

    Parameters
    ----------
    a: np.darray
        Dimension array.
    b: np.darray
        Benchmark array.

    Returns
    -------
    a_dless: np.darray
        Dimensionless array.
    """
    a_dless = a / b

    return a_dless

###############################################################################
#                              Restore dimension                              #
###############################################################################
# @njit(fastmath=False)
def dimension(a_dless, b):
    """Solve the dimension array.

    Parameters
    ----------
    a_dless: np.darray
        Dimensionless array.
    b: np.darray
        Benchmark array.

    Returns
    -------
    a_dim: np.darray
        Dimension array.
    """
    a_dim = a_dless * b

    return a_dim

###############################################################################
#                         Calculate no load position                          #
###############################################################################
# @numba.njit(fastmath=False)
def no_load_position(Info_es, mod_nlp):
    """Store the results.

    Parameters
    ----------
    Info_es: tuple
        Information of no_load_position.
    mod_nlp: tuple
        Module of no_load_position.

    Returns
    -------
    x_no_load: np.darray
        No load position of component.
    Info_nlp: tuple
        Information of no_load_position.
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    (D_b,
     D_m,
     F_x,
     F_y,
     F_z,
     R_b,
     Shim_thknss_i,
     Shim_thknss_o,
     f_i,
     f_o,
     free_con_ang,
     k_geo_imc_type_i,
     k_geo_imc_type_o,
     mis_i_y,
     mis_i_z,
     mis_o_y,
     mis_o_z,
     n,
     var_i_r0,
     var_i_r1,
     var_i_r2,
     var_o_r0,
     var_o_r1,
     var_o_r2,
     shim_ang_o) = mod_nlp[0::]

    P_rad_o = 0.5 * (D_m - 2 * (f_o - 0.5) * D_b * math.cos(free_con_ang))
    P_rad_i = 0.5 * (D_m + 2 * (f_i - 0.5) * D_b * math.cos(free_con_ang))
    P_rad_o += Info_es[9]
    P_rad_i += Info_es[10]
    ###########################################################################
    #                     Outer race initial misalignment                     #
    ###########################################################################
    T_omis_o = np.zeros((1, 3, 3))
    T_omis_o[0, :, :] = np.identity(3)
    T_I_omis = np.zeros((1, 3, 3))
    T_I_omis[0, 0, 0] = math.cos(mis_o_y) * math.cos(mis_o_z)
    T_I_omis[0, 0, 1] = math.sin(mis_o_z)
    T_I_omis[0, 0, 2] = -math.sin(mis_o_y) * math.cos(mis_o_z)
    T_I_omis[0, 1, 0] = -math.cos(mis_o_y) * math.sin(mis_o_z)
    T_I_omis[0, 1, 1] = math.cos(mis_o_z)
    T_I_omis[0, 1, 2] = math.sin(mis_o_y) * math.sin(mis_o_z)
    T_I_omis[0, 2, 0] = math.sin(mis_o_y)
    T_I_omis[0, 2, 2] = math.cos(mis_o_y)
    ###########################################################################
    #                     Inner race initial misalignment                     #
    ###########################################################################
    T_imis_i = np.zeros((1, 3, 3))
    T_imis_i[0, :, :] = np.identity(3)
    T_I_imis = np.zeros((1, 3, 3))
    T_I_imis[0, 0, 0] = math.cos(mis_i_y) * math.cos(mis_i_z)
    T_I_imis[0, 0, 1] = math.sin(mis_i_z)
    T_I_imis[0, 0, 2] = -math.sin(mis_i_y) * math.cos(mis_i_z)
    T_I_imis[0, 1, 0] = -math.cos(mis_i_y) * math.sin(mis_i_z)
    T_I_imis[0, 1, 1] = math.cos(mis_i_z)
    T_I_imis[0, 1, 2] = math.sin(mis_i_y) * math.sin(mis_i_z)
    T_I_imis[0, 2, 0] = math.sin(mis_i_y)
    T_I_imis[0, 2, 2] = math.cos(mis_i_y)
    ###########################################################################
    #                               Ball azimuth                              #
    ###########################################################################
    dsi = 2 * math.pi / n
    phi_b = np.zeros((n, 1, 1))
    T_I_a = np.zeros((n, 3, 3))
    T_I_a[:,0,0] = 1.
    for i in range(n):
        phi_b[i, 0, 0] = i * dsi
        T_I_a[i, 1, 1] = math.cos(phi_b[i, 0, 0])
        T_I_a[i, 1, 2] = math.sin(phi_b[i, 0, 0])
        T_I_a[i, 2, 1] = -T_I_a[i, 1, 2]
        T_I_a[i, 2, 2] = T_I_a[i, 1, 1]

    theta = math.atan2(-F_y, F_z)

    t0 = np.zeros((1, 3, 3))
    t0[:, 0, 0] = 1
    t0[:, 1, 1] = math.cos(theta)
    t0[:, 1, 2] = math.sin(theta)
    t0[:, 2, 1] = -t0[:, 1, 2]
    t0[:, 2, 2] = t0[:, 1, 1]

    e_i = race_radius(
        0, P_rad_i, phi_b, k_geo_imc_type_i, var_i_r0, var_i_r1, var_i_r2
    )
    e_o = race_radius(
        0, P_rad_o, phi_b, k_geo_imc_type_o, var_o_r0, var_o_r1, var_o_r2
    )

    rc = np.zeros((1, 3, 1))
    rc[0, 0, 0] = 0.5 * (Shim_thknss_i - Shim_thknss_o)
    rc[0, 2, 0] = e_i[0, 0, 0] - e_o[0, 0, 0]

    aa = f_o * D_b + f_i * D_b - D_b
    if aa < rc[0, 2, 0]:
        ax = -rc[0, 0, 0]
        ar = 0.
    elif F_x != 0:
        ax = math.sqrt(aa ** 2 - rc[0, 2, 0] ** 2) - rc[0, 0, 0]
        ar = 0
    elif F_y != 0 or F_z != 0:
        ar = math.sqrt(aa ** 2 - rc[0, 0, 0] ** 2) - rc[0, 2, 0]
        ax = 0.
    ###########################################################################
    #                       Set no load race positions                        #
    ###########################################################################
    r_ig_I = np.zeros((1, 3, 1))
    r_ig_I[0, 0, 0] = ax
    r_ig_I[0, 1, 0] = ar * t0[0, 2, 1]
    r_ig_I[0, 2, 0] = ar * t0[0, 2, 2]

    r_ogc_o_a = np.zeros((n, 3, 1))
    r_ogc_o_a[:, 0, 0] = -0.5 * Shim_thknss_o
    r_ogc_o_a[:, 2, 0] = e_o[:, 0, 0]

    r_igc_i_a = np.zeros((n, 3, 1))
    r_igc_i_a[:, 0, 0] = 0.5 * Shim_thknss_i
    r_igc_i_a[:, 2, 0] = e_i[:, 0, 0]

    r_ig_a = np.zeros((n, 3, 1))
    t0 = np.zeros((3, 3))
    t1 = np.zeros((3, 1))
    for i in range(n):
        t0[:, :] = T_I_a[i, :, :]
        t1[:, :] = r_ig_I[0, :, :]
        t2 = np.dot(t0, t1)
        r_ig_a[i, :, :] = t2

    r_igc_i_a += r_ig_a

    rc = r_igc_i_a - r_ogc_o_a

    alfa = math.atan(rc[0, 0, 0] / rc[0, 2, 0])
    sa = math.sin(alfa)
    ca = math.cos(alfa)

    ar = f_o * D_b - R_b
    ###########################################################################
    #                       Set no load ball positions                        #
    ###########################################################################
    r_bg_I = np.zeros((n, 3, 1))
    if ax == 0:
        r_bg_I[:, 0, 0] = 0.
        r_bg_I[:, 1, 0] = r_ogc_o_a[:, 2, 0] + ar * math.cos(shim_ang_o)
        r_bg_I[:, 2, 0] = phi_b[:, 0, 0]
    else:
        r_bg_I[:, 0, 0] = r_ogc_o_a[:, 0, 0] + ar * sa
        r_bg_I[:, 1, 0] = r_ogc_o_a[:, 2, 0] + ar * ca
        r_bg_I[:, 2, 0] = phi_b[:, 0, 0]
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    x_no_load = np.zeros(37+12*n)
    x_no_load[0] = r_ig_I[0, 0, 0]
    x_no_load[1] = r_ig_I[0, 1, 0]
    x_no_load[2] = r_ig_I[0, 2, 0]
    x_no_load[24:24+12*n:12] = r_bg_I[:, 0, 0]
    x_no_load[26:24+12*n:12] = r_bg_I[:, 1, 0]
    x_no_load[28:24+12*n:12] = r_bg_I[:, 2, 0]

    Info_nlp = (P_rad_o,
                # Race pressure radius of inner race.
                P_rad_i
                # Race pressure radius of inner race.
                )

    return x_no_load, Info_nlp

###############################################################################
#                         Calculate initial position                          #
###############################################################################
# @numba.njit(fastmath=False)
def initial_position(x_no_load, Info_nlp, Info_es, mod_ip, mod_brcs, mod_brcf):
    """Solve stle lubrication factor.

    Parameters
    ----------
    x_no_load: np.darray
        No load position of component.
    Info_nlp: tuple
        Information of no_load_position.
    Info_es: tuple
        Information of expansion_subcall.
    mod_ip: tuple
        Module of initial_pos.
    mod_brcs: tuple
        Module of ball_race_contact_strain.
    mod_brcf: tuple
        Module of ball_race_contact_force.

    Returns
    -------
    x_init: np.darray
        Initial position of component.
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    (F_x,
     F_y,
     F_z,
     R_b,
     f_i,
     f_o,
     free_con_ang,
     n,
     rpm_i,
     rpm_o) = mod_ip[0::]

    P_rad_o, P_rad_i = Info_nlp[0::]
    ###########################################################################
    #                       Initial stiffness position                        #
    ###########################################################################
    x_stiff = np.zeros(37+12*n)
    x_stiff = np.copy(x_no_load)
    x_stiff[24:24+12*n:12] += 1e-3 * R_b
    ###########################################################################
    #                           Initial displayment                           #
    ###########################################################################
    F_norm = math.sqrt(F_y ** 2 + F_z ** 2)
    ###########################################################################
    #                     Axial load acted on the bearing                     #
    ###########################################################################
    if F_x != 0:
        delta_b_o, delta_b_i, Info_brcs =  ball_race_contact_strain(
            x_stiff, Info_es, mod_brcs
        )

        Q_b_o, Q_b_i, Info_brcf = ball_race_contact_force(
            delta_b_o, delta_b_i, mod_brcf
        )

        k_b_o = Q_b_o[0, 0, 0] / delta_b_o[0, 0, 0]

        re_dis_x = 1e-3 * R_b + F_x / n / k_b_o
        re_dis_r = 0.

        race_dis_x = 2 * re_dis_x
        race_dis_r = 0.
    ###########################################################################
    #                  Only radial load acted on the bearing                  #
    ###########################################################################
    elif F_y != 0 or F_z != 0:
        x_stiff[26] = x_no_load[26] + 1e-3 * R_b

        delta_b_o, delta_b_i, Info_brcs =  ball_race_contact_strain(
            x_stiff, Info_es, mod_brcs
        )

        Q_b_o, Q_b_i, Info_brcf = ball_race_contact_force(
            delta_b_o, delta_b_i, mod_brcf
        )

        Q_b_o_max = np.max(Q_b_o[:, 0, 0])
        delta_b_o_max = np.max(delta_b_o[:, 0, 0])

        k_b_o = 1.5 * Q_b_o_max / delta_b_o_max

        re_dis_x = F_norm / k_b_o
        re_dis_r = 0.

        race_dis_x = 0.
        race_dis_r = 2 * re_dis_r
    ###########################################################################
    #                       Set initial race positions                        #
    ###########################################################################
    F_norm = math.sqrt(F_y ** 2 + F_z ** 2)
    if F_norm == 0:
        cth, sth = 1., 0.
    else:
        cth, sth = F_z / F_norm, -F_y / F_norm

    x_init = np.copy(x_no_load)
    x_init[0] = x_no_load[0] + race_dis_x
    x_init[2] = x_no_load[2] - race_dis_r * sth
    x_init[4] = x_no_load[4]  + race_dis_r * cth

    omg_i = rpm_i * math.pi / 30
    omg_o = rpm_o * math.pi / 30

    x_init[7] = omg_i
    x_init[31+12*n] = omg_o
    """
    ###########################################################################
    #                         Set initial ball speed                          #
    ###########################################################################
    con_ang = free_con_ang
    omg_o = rpm_o * math.pi / 30
    omg_i = rpm_i * math.pi / 30

    X = x_stiff[26] + 0.5 * D_b * math.cos(con_ang)
    Y = x_stiff[26] - 0.5 * D_b * math.cos(con_ang)

    A = np.zeros((3, 3))
    A[0, 0] = math.sin(con_ang)
    A[0, 1] = math.cos(con_ang)
    A[0, 2] = math.sin(con_ang)
    A[1, 0] = R_b * math.cos(con_ang)
    A[1, 1] = -R_b * math.sin(con_ang)
    A[1, 2] = X
    A[2, 0] = R_b * math.cos(con_ang)
    A[2, 1] = -R_b * math.sin(con_ang)
    A[2, 2] = Y

    B = np.zeros((3, 1))
    B[0, 0] = omg_o * math.sin(con_ang)
    B[1, 0] = omg_o * X
    B[2, 0] = omg_i * Y

    C = np.linalg.solve(A, B)
    """
    ###########################################################################
    #                       Set initial ball positions                        #
    ###########################################################################
    ssi = np.zeros(n)
    ssi[:,] = np.sin(x_no_load[28:24+12*n:12])

    csi = np.zeros(n)
    csi[:,] = np.cos(x_no_load[28:24+12*n:12])

    x_init[24:24+12*n:12] = x_init[24:24+12*n:12] + re_dis_x
    x_init[26:24+12*n:12] = (
        x_init[26:24+12*n:12] + re_dis_r * (csi[:,] * cth + ssi[:,] * sth)
    )
    x_init[28:24+12*n:12] = x_no_load[28:24+12*n:12]

    # x_init[29:24+12*n:12] = 0.5 * (omg_o + omg_i)  # C[2, 0]
    # x_init[31:24+12*n:12] = C[0, 0]
    # x_init[35:24+12*n:12] = C[1, 0]

    # x_init[19] = np.mean(x_init[29:24+12*n:12])

    return x_init

###############################################################################
#                   Vector transformation and bearing equations               #
###############################################################################
def quasi_static_for_ball_bearing(
    x, forc_and_momt, cont_swit, mod_type, prop_b, prop_i, dim_para, oth_para
    ):
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    F_x, F_y, F_z, M_y, M_z = forc_and_momt[0::]

    k_brg_mov, k_tra_i_x, k_tra_i_y, k_tra_i_z = cont_swit[0::]

    (mod_tc, mod_es, mod_brcs, mod_brcf, mod_brcs_,
     mod_brcf_, mod_cf, mod_mff, mod_bbf) = mod_type[0::]

    m_b = prop_b[0]
    m_i = prop_i[0]

    s, s_load = dim_para[0::]

    x_ref, max_rs, n, n_cseg = oth_para[0::]

    if len(x) <= 3:
        x_ref[0] = x[-3]
        x_ref[2] = x[-2]
        x_ref[4] = x[-1]
    elif len(x) == (2 * n):
        x_ref[24:24+12*n:12] = x[0:2*n:2]
        x_ref[26:24+12*n:12] = x[1:2*n:2]
    else:
        x_ref[0] = x[-3]
        x_ref[2] = x[-2]
        x_ref[4] = x[-1]
        x_ref[24:24+12*n:12] = x[0:2*n:2]
        x_ref[26:24+12*n:12] = x[1:2*n:2]

    x_dim = dimension(x_ref, s)
    ###########################################################################
    #                           Temperature change                            #
    ###########################################################################
    Info_tc = mod_tc
    """
    Temperature change need to be development.
    else:
        Info_tc = (temp_o, temp_i, temp_h, temp_s, temp_c, temp_r)
    """
    ###########################################################################
    #                  Solve the clearance and size change                    #
    ###########################################################################
    Info_es = expansion_subcall(x_dim, Info_tc, mod_es)
    ###########################################################################
    #                          Solve each component                           #
    ###########################################################################
    delta_b_o, delta_b_i, Info_brcs = ball_race_contact_strain(
        x_dim, Info_es, mod_brcs
    )

    Q_b_o, Q_b_i, Info_brcf = ball_race_contact_force(
        delta_b_o, delta_b_i, mod_brcf
    )

    if max_rs == 2:
        delta_b_o_, delta_b_i_ = np.zeros((n, 1, 1)), np.zeros((n, 1, 1))
        Info_brcs_ = (
            np.zeros((n, 3, 3)), np.zeros((n, 3, 3)), np.zeros((n, 1, 1)),
            np.zeros((n, 1, 1)), np.zeros((n, 1, 1)), np.zeros((n, 1, 1)),
            np.zeros((n, 1, 1)), np.zeros((n, 1, 1)), 0., 0.
        )

        Q_b_o_, Q_b_i_ = np.zeros((n, 1, 1)), np.zeros((n, 1, 1))
        Info_brcf_ = (
            np.zeros((n, 1, 1)), np.zeros((n, 1, 1)), np.zeros((n, 1, 1)),
            np.zeros((n, 1, 1)), np.zeros((n, 1, 1)), np.zeros((n, 1, 1)),
            np.zeros((n, 1, 1)), np.zeros((n, 1, 1)), np.zeros((n, 1, 1)),
            np.zeros((n, 1, 1)), np.zeros((n, 1, 1)), np.zeros((n, 1, 1))
        )
    else:
        delta_b_o_, delta_b_i_, Info_brcs_ = ball_race_contact_strain_(
            Info_brcs, mod_brcs_
        )

        Q_b_o_, Q_b_i_, Info_brcf_ = ball_race_contact_force_(
            delta_b_o_, delta_b_i_, Info_brcs_, mod_brcf_
        )

    C, F_b_cff, F_b_gyro, G_b_gyro = ball_centrifugal_forece(
        x_dim, Info_brcs, Info_brcf, mod_cf
    )

    """
    Need to be developed.
    """
    if n_cseg <= 0:
        F_b_b, M_b_b, Info_bbf = ball_ball_force(
            x_dim, Info_brcs, Info_brtf, mod_bbf
        )
    else:
        F_b_b = (np.zeros((n, 3, 1)),)
        M_b_b = (np.zeros((n, 3, 1)),)
        Info_bbf = (
            np.zeros((n, 1, 1)), np.zeros((n, 1, 1)),
            np.zeros((n, 3, 1)), np.zeros((n, 3, 1)),
            np.zeros((n, 1, 1))
        )
    ###########################################################################
    #             Calculate moving bearing reference frame force              #
    ###########################################################################
    if k_brg_mov <= 0:
        k0, k1, k2 = 0, 0, 0
        F_b_mov_a, F_i_mov_I = np.zeros((n, 3, 1)), np.zeros((1, 3, 1))
    else:
        (R_brg_orb, brg_ang_vel, brg_load_frac_i, brg_load_frac_o,
         brg_mov_dir, m_b, m_c, m_i, m_o, n, n_cseg) = mod_mff[0::]
        r_bg_I = Info_brcs[15]

        if brg_mov_dir == 0:
            k0, k1, k2 = 0, 0, 1

            F_b_mov_x_I = np.zeros((n, 1, 1))

            F_b_mov_y_I = np.zeros((n, 1, 1))
            F_b_mov_y_I[:, 0, 0] = (
                2 * brg_ang_vel * r_bg_I[:, 1, 0] * C[:, 2, 0] +
                r_bg_I[:, 1, 0] * brg_ang_vel ** 2
            ) * m_b

            F_b_mov_z_I = np.zeros((n, 1, 1))
            F_b_mov_z_I[:, 0, 0] = (
                2 * brg_ang_vel * r_bg_I[:, 2, 0] * C[:, 2, 0] +
                (r_bg_I[:, 2, 0] + R_brg_orb) * brg_ang_vel ** 2
            ) * m_b

            F_b_mov_a = np.zeros((n, 3, 1))
            F_b_mov_a[:, 0, 0] = F_b_mov_x_I[:, 0, 0]
            F_b_mov_a[:, 2, 0] = (
                -F_b_mov_y_I[:, 0, 0] * np.sin(x_dim[28:24+12*n:12]) +
                F_b_mov_z_I[:, 0, 0] * np.cos(x_dim[28:24+12*n:12])
            )
        elif brg_mov_dir == 1:
            k0, k1, k2 = 0, 1, 0

            F_b_mov_x_I = np.zeros((n, 1, 1))
            F_b_mov_x_I[:, 0, 0] = (
                2 * brg_ang_vel * r_bg_I[:, 1, 0] * C[:, 2, 0] +
                (r_bg_I[:, 0, 0] + R_brg_orb) * brg_ang_vel ** 2
            ) * m_b

            F_b_mov_y_I = np.zeros((n, 1, 1))
            F_b_mov_y_I[:, 0, 0] = r_bg_I[:, 2, 0] * brg_ang_vel ** 2 * m_b

            F_b_mov_a = np.zeros((n, 3, 1))
            F_b_mov_a[:, 0, 0] = F_b_mov_x_I[:, 0, 0]
            F_b_mov_a[:, 2, 0] = (
                F_b_mov_y_I[:, 0, 0] * np.cos(x_dim[28:24+12*n:12])
            )
        elif brg_mov_dir == 2:
            k0, k1, k2 = 1, 0, 0

            F_b_mov_x_I = np.zeros((n, 1, 1))
            F_b_mov_x_I[:, 0, 0] = (
                2 * brg_ang_vel * r_bg_I[:, 2, 0] * C[:, 2, 0] +
                (r_bg_I[:, 0, 0] + R_brg_orb) * brg_ang_vel ** 2
            ) * m_b
            F_b_mov_y_I = np.zeros((n, 1, 1))
            F_b_mov_y_I[:, 0, 0] = r_bg_I[:, 1, 0] * brg_ang_vel ** 2 * m_b

            F_b_mov_a = np.zeros((n, 3, 1))
            F_b_mov_a[:, 0, 0] = F_b_mov_x_I[:, 0, 0]
            F_b_mov_a[:, 2, 0] = (
                F_b_mov_y_I[:, 0, 0] * np.cos(x_dim[28:24+12*n:12])
            )

        F_i_mov = (
            (m_i * brg_load_frac_i -
             m_o * brg_load_frac_o) *
            brg_ang_vel ** 2 * R_brg_orb
        )

        F_i_mov_I = np.zeros((1, 3, 1))
        F_i_mov_I[0, 0, 0] = k0 * F_i_mov
        F_i_mov_I[0, 1, 0] = k1 * F_i_mov
        F_i_mov_I[0, 2, 0] = k2 * F_i_mov
    ###########################################################################
    #                             Sum component                               #
    ###########################################################################
    phi_b_i, alpha_b_o_0, alpha_b_i_0 = (
        Info_brcs[18], Info_brcs[20], Info_brcs[22]
    )
    alpha_b_o_0_, alpha_b_i_0_ = Info_brcs_[4], Info_brcs_[6]

    sin_alpha_b_o_0 = np.sin(alpha_b_o_0)
    cos_alpha_b_o_0 = np.cos(alpha_b_o_0)

    sin_alpha_b_i_0 = np.sin(alpha_b_i_0)
    cos_alpha_b_i_0 = np.cos(alpha_b_i_0)

    sin_alpha_b_o_0_ = np.sin(alpha_b_o_0_)
    cos_alpha_b_o_0_ = np.cos(alpha_b_o_0_)

    sin_alpha_b_i_0_ = np.sin(alpha_b_i_0_)
    cos_alpha_b_i_0_ = np.cos(alpha_b_i_0_)

    FF_i_X = (k_tra_i_x * F_x + k0 * F_i_mov_I[0, 0, 0] +
              np.sum(Q_b_i[:, 0, 0] * sin_alpha_b_i_0[:, 0, 0] +
                     Q_b_i_[:, 0, 0] * sin_alpha_b_i_0_[:, 0, 0]))

    FF_i_Y = (k_tra_i_y * F_y - k1 * F_i_mov_I[0, 1, 0] -
              np.sum((Q_b_i[:, 0, 0] * cos_alpha_b_i_0[:, 0, 0] +
                      Q_b_i_[:, 0, 0] * cos_alpha_b_i_0_[:, 0, 0]) *
                      np.sin(phi_b_i[:, 0, 0])))

    FF_i_Z = (k_tra_i_z * F_z + k2 * F_i_mov_I[0, 2, 0] +
              np.sum((Q_b_i[:, 0, 0] * cos_alpha_b_i_0[:, 0, 0] +
                      Q_b_i_[:, 0, 0] * cos_alpha_b_i_0_[:, 0, 0]) *
                      np.cos(phi_b_i[:, 0, 0])))

    FF_b_X = np.zeros((n, 1, 1))
    FF_b_X[:, 0, 0] = (Q_b_o[:, 0, 0] * sin_alpha_b_o_0[:, 0, 0] +
                       Q_b_i[:, 0, 0] * sin_alpha_b_i_0[:, 0, 0] +
                       Q_b_o_[:, 0, 0] * sin_alpha_b_o_0_[:, 0, 0] +
                       Q_b_i_[:, 0, 0] * sin_alpha_b_i_0_[:, 0, 0])
    FF_b_X[:, 0, 0] += (-F_b_mov_a[:, 0, 0] -
                        F_b_gyro[:, 0, 0] * cos_alpha_b_o_0[:, 0, 0] -
                        F_b_gyro[:, 0, 1] * cos_alpha_b_i_0[:, 0, 0])

    FF_b_Z = np.zeros((n, 1, 1))
    FF_b_Z[:, 0, 0] = (Q_b_o[:, 0, 0] * cos_alpha_b_o_0[:, 0, 0] +
                       Q_b_i[:, 0, 0] * cos_alpha_b_i_0[:, 0, 0] +
                       Q_b_o_[:, 0, 0] * cos_alpha_b_o_0_[:, 0, 0] +
                       Q_b_i_[:, 0, 0] * cos_alpha_b_i_0_[:, 0, 0])
    FF_b_Z[:, 0, 0] += (-F_b_cff[:, 0, 0] - F_b_mov_a[:, 2, 0] +
                        F_b_gyro[:, 0, 0] * sin_alpha_b_o_0[:, 0, 0] +
                        F_b_gyro[:, 0, 1] * sin_alpha_b_i_0[:, 0, 0])
    ###########################################################################
    #                            Dynamic equations                            #
    ###########################################################################
    g = np.zeros(len(x))
    if len(x) <= 3:
        g[-3] = m_b / m_i * FF_i_X / s_load
        g[-2] = m_b / m_i * FF_i_Y / s_load
        g[-1] = m_b / m_i * FF_i_Z / s_load
    elif len(x) == (2 * n):
        g[0:2*n:2] = FF_b_X[:, 0, 0] / s_load
        g[1:2*n:2] = FF_b_Z[:, 0, 0] / s_load
    else:
        g[-3] = m_b / m_i * FF_i_X / s_load
        g[-2] = m_b / m_i * FF_i_Y / s_load
        g[-1] = m_b / m_i * FF_i_Z / s_load
        g[0:2*n:2] = FF_b_X[:, 0, 0] / s_load
        g[1:2*n:2] = FF_b_Z[:, 0, 0] / s_load

    return g


if __name__=="__main__":
    ###########################################################################
    #                             Read base data                              #
    ###########################################################################
    with open('Input.pickle', 'rb') as f:
        input_data = pickle.load(f)

    brg_type = 0
    base_data = input_data[0][1::]
    for i in range(0, len(base_data)):
        list_base_data = list(base_data[i].keys())
        for j in range(len(list_base_data)):
            globals()[list_base_data[j]] = base_data[i][list_base_data[j]]
    ###########################################################################
    #                      Solve preparation parameters                       #
    ###########################################################################
    tj = special.roots_legendre(4)[0].reshape((1, 1, 4))
    hj = special.roots_legendre(4)[1].reshape((1, 1, 4))
    tj8, hj8 = special.roots_legendre(8)
    tj10, hj10 = special.roots_legendre(10)

    Info_ec = expansion_constant(
        den_h, E_h, po_h, D_h, D_o_u,
        den_s, E_s, po_s, D_i_d, D_s,
        den_o, E_o, po_o, D_o_d,
        den_i, E_i, po_i, D_i_u,
        den_c, E_c, po_c, D_c_u, D_c_d,
        coeff_ther_exp_h, coeff_ther_exp_s,
        coeff_ther_exp_o, coeff_ther_exp_i,
        coeff_ther_exp_c, n_cseg
    )

    (vh00, vh01, vh02, vh03, vs00, vs01, vs02, vs03,
     vo00, vo01, vo02, vo03, vo10, vo11, vo12, vo13,
     vi00, vi01, vi02, vi03, vi10, vi11, vi12, vi13,
     vc00, vc01, vc02, vc03, vc10, vc11, vc12, vc13,
     sth00, sth01, sth02, sts00, sts01, sts02,
     sto00, sto01, sto02, sto10, sto11, sto12,
     sti00, sti01, sti02, sti10, sti11, sti12,
     stc00, stc01, stc02, stc10, stc11, stc12) = Info_ec

    max_rs = 2
    if brg_type == 0:
        if Shim_thknss_o > 0 or Shim_thknss_i > 0:
            max_rs = 4
    ###########################################################################
    #                         Solve contact stiffness                         #
    ###########################################################################
    cos_ang = math.cos(free_con_ang)
    d_m_over_cos = D_m / cos_ang

    r0x_bo = -(d_m_over_cos + D_b) * 0.5
    r0y_bo = -f_o * D_b

    r0x_bi = (d_m_over_cos - D_b) * 0.5
    r0y_bi = -f_i * D_b

    r1x_bo = r1y_bo = R_b
    r1x_bi = r1y_bi = R_b

    r0x_list = [r0x_bo, r0x_bi]
    r0y_list = [r0y_bo, r0y_bi]

    r1x_list = [r1x_bo, r1x_bi]
    r1y_list = [r1y_bo, r1y_bi]

    E0_list = [E_o, E_i]
    E1_list = [E_b for _ in range(2)]

    po0_list = [po_o, po_i]
    po1_list = [po_b for _ in range(2)]

    stype_list = [0 for _ in range(2)]

    Info_hs = tuple([hertz_stiffness(
        r0x, r0y, r1x, r1y, E0, po0, E1, po1, stype
        ) for r0x, r0y, r1x, r1y, E0, po0, E1, po1, stype in zip(
            r0x_list, r0y_list, r1x_list, r1y_list, E0_list,
            po0_list, E1_list, po1_list, stype_list
        )
    ])

    (K_b_o, ke_b_o, de_b_o, t_b_o, E_b_o, ep_b_o,
     R_yipu_b_o, R_b_o, a_b_o, b_b_o) = Info_hs[0]

    (K_b_i, ke_b_i, de_b_i, t_b_i, E_b_i, ep_b_i,
     R_yipu_b_i, R_b_i, a_b_i, b_b_i) = Info_hs[1]

    ee_b_o  = special.ellipe(1 / ke_b_o)
    ee_b_i  = special.ellipe(1 / ke_b_i)
    ###########################################################################
    #                          Solve traction curve                           #
    ###########################################################################
    A_0, B_0, C_0, D_0 = traction_coefficient(kai0_0, kaiin_0, kaim_0, um_0)
    ###########################################################################
    #                         Solve drag coefficient                          #
    ###########################################################################
    """
    if k_chrn_type > 0:
        f_drag = drag_coefficient()
    """
    ###########################################################################
    #                       Solve initial misalignment                        #
    ###########################################################################
    T_I_omis, T_I_imis = misalignment(mis_o_y, mis_o_z, mis_i_y, mis_i_z)
    ###########################################################################
    #                        Solve irregular geometry                         #
    ###########################################################################
    r_bg_bm_b, T_b_bp, T_bp_b = geometry_eccentric(
        geo_cen_b_x, geo_cen_b_y, geo_cen_b_z,
        deg_b_x, deg_b_y, deg_b_z,
        n, var_inr_b, var_inr_b
    )
    ###########################################################################
    #                 Initial seting for non-linear equation                  #
    ###########################################################################
    a0 = np.zeros((37+12*n))
    a0[7] = rpm_i * math.pi / 30
    a0[31+12*n] = rpm_o * math.pi / 30
    a0[19] = 0.5 * (a0[7] + a0[31+12*n])

    s_len = R_b
    s_load = max(abs(F_x), abs(F_y), abs(F_z))
    s_time = math.sqrt(m_b * R_b / s_load)
    s_vel = 1 / s_time * s_len
    s_angv = 1 / s_time

    s = np.ones_like(a0)

    ass_rules = {
        0: (s_len, 2), 1: (s_vel, 2), 7: (s_angv, 3)
    }

    for ofs, (value, num_ofss) in ass_rules.items():
        for i in range(num_ofss):
            curr_ofs = ofs + 2 * i
            if curr_ofs <= 11:
                s[curr_ofs::12] = value

    s[[4, 16, 28+12*n]] = s_len
    s[[5, 17, 29+12*n]] = s_vel
    s[slice(29, 24+12*n, 12)] = s_angv
    s[-1] = s_time
    ###########################################################################
    #                          Module in the subroutine                       #
    ###########################################################################
    mod_index_data = input_data[1]
    mod_list_name = mod_index_data[0]
    for i in range(2, len(mod_list_name) + 2):
        globals()[mod_list_name[i-2]] = []
        for j in range(len(mod_index_data[i])):
            if mod_index_data[i][j] in locals():
                value = locals()[mod_index_data[i][j]]
            elif mod_index_data[i][j] in globals():
                value = globals()[mod_index_data[i][j]]
            else:
                value = None
            if value is not None:
                globals()[mod_list_name[i-2]].append(value)
    for i in range(2, len(mod_list_name) + 2):
        globals()[mod_list_name[i-2]] = tuple(globals()[mod_list_name[i-2]])
    ###########################################################################
    #                Solve custom variable loads and moments                  #
    ###########################################################################
    Info_udfc = ()  # user_defined_force_curve(time_step_number, 'all')
    ###########################################################################
    #                          Fatigue life constant                          #
    ###########################################################################
    if k_life_freq > 0:
        Info_flc = fatigue_life_constant(mod_flc)
    ###########################################################################
    #                        Delete other file in path                        #
    ###########################################################################
    # file_name = ['Input.pickle', 'Initial_value.npy']
    # for i in range(len(file_name)):
        # os.remove(file_name[i])
    ###########################################################################
    #                              Dimensionless                              #
    ###########################################################################
    Info_tc = mod_tc
    Info_es = expansion_subcall(a0, Info_tc, mod_es)

    x_no_load, Info_nlp = no_load_position(Info_es, mod_nlp)
    x_guess = initial_position(
        x_no_load, Info_nlp, Info_es, mod_ip, mod_brcs, mod_brcf
    )

    a0 = dimensionless(x_guess, s)
    ###########################################################################
    #                                Main loop                                #
    ###########################################################################
    x_base = np.copy(a0)

    x_init = np.zeros(2*n+3)
    x_init[0:2*n:2] = x_base[24:24+12*n:12]
    x_init[1:2*n:2] = x_base[26:24+12*n:12]
    x_init[-3] = x_base[0]
    x_init[-2] = x_base[2]
    x_init[-1] = x_base[4]

    forc_and_momt = (F_x, F_y, F_z, M_y, M_z)

    cont_swit = (k_brg_mov, k_tra_i_x, k_tra_i_y, k_tra_i_z)

    mod_type = (
        mod_tc, mod_es, mod_brcs, mod_brcf,
        mod_brcs_, mod_brcf_, mod_cf, mod_mff, mod_bbf
    )

    prop_b = (m_b,)
    prop_i = (m_i,)

    dim_para = (s, s_load)

    oth_para = (x_base, max_rs, n, n_cseg)

    args = (
        forc_and_momt, cont_swit, mod_type,
        prop_b, prop_i, dim_para, oth_para
    )

    result = opt.least_squares(
        quasi_static_for_ball_bearing,
        x_init, method='dogbox', verbose=1, args=args,
        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=200
    )

    if result.cost >= 1e-12:
        result = opt.least_squares(
            quasi_static_for_ball_bearing,
            result.x, method='trf', verbose=1, args=args,
            ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=800
        )
        #######################################################################
        #                        Alternating solution                         #
        #######################################################################
        if result.cost >= 1e-12:
            x_base = np.copy(a0)
            for i in range(100):
                x_init = np.zeros(3)
                x_init[0] = x_base[0]
                x_init[1] = x_base[2]
                x_init[2] = x_base[4]

                oth_para = (x_base, max_rs, n, n_cseg)

                args = (
                    forc_and_momt, cont_swit, mod_type,
                    prop_b, prop_i, dim_para, oth_para
                )

                result_r = opt.least_squares(
                    quasi_static_for_ball_bearing,
                    x_init, method='dogbox', verbose=0, args=args,
                    ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=100
                )
                if result_r.cost >= 1e-12:
                    result_r = opt.least_squares(
                        quasi_static_for_ball_bearing,
                        result_r.x, method='trf', verbose=0, args=args,
                        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=200
                    )

                x_base[0] = result_r.x[0]
                x_base[2] = result_r.x[1]
                x_base[4] = result_r.x[2]

                x_init = np.zeros(2*n)
                x_init[0:2*n:2] = x_base[24:24+12*n:12]
                x_init[1:2*n:2] = x_base[26:24+12*n:12]

                oth_para = (x_base, max_rs, n, n_cseg)

                args = (
                    forc_and_momt, cont_swit, mod_type,
                    prop_b, prop_i, dim_para, oth_para
                )

                result_b = opt.least_squares(
                    quasi_static_for_ball_bearing,
                    x_init, method='dogbox', verbose=0, args=args,
                    ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=100
                )
                if result_b.cost >= 1e-12:
                    result_b = opt.least_squares(
                        quasi_static_for_ball_bearing,
                        result_b.x, method='trf', verbose=0, args=args,
                        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=200
                    )

                x_base[24:24+12*n:12] = result_b.x[0:2*n:2]
                x_base[26:24+12*n:12] = result_b.x[1:2*n:2]
                ###############################################################
                #                         Check error                         #
                ###############################################################
                x_init = np.zeros(3)
                x_init[0] = x_base[0]
                x_init[1] = x_base[2]
                x_init[2] = x_base[4]

                oth_para = (x_base, max_rs, n, n_cseg)

                args = (
                    forc_and_momt, cont_swit, mod_type,
                    prop_b, prop_i, dim_para, oth_para
                )

                result_r = opt.least_squares(
                    quasi_static_for_ball_bearing,
                    x_init, method='dogbox', verbose=0, args=args,
                    ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=1
                )

                costs = [result_r.cost, result_b.cost]
                if all(cost < 1e-12 for cost in costs) or i >= 99:
                    result.x[-3::] = result_r.x[0::]
                    result.x[0:2*n] = result_b.x[0::]
                    ###########################################################
                    #            Renew all positions of components            #
                    ###########################################################
                    result = opt.least_squares(
                        quasi_static_for_ball_bearing,
                        result.x, method='dogbox', verbose=1, args=args,
                        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=200
                    )
                    break
    ###########################################################################
    #                           Store total results                           #
    ###########################################################################
    x = np.copy(x_guess)
    x[0:6:2] = result.x[-3::] * s_len
    x[7] = rpm_i * math.pi / 30
    x[31+12*n] = rpm_o * math.pi / 30
    x[24:24+12*n:12] = result.x[0:2*n:2] * s_len
    x[26:24+12*n:12] = result.x[1:2*n:2] * s_len
    x[12] = cage_mass_cen_x
    x[12] += np.average(x[24:24+12*n:12]) * s_len
    x[14] = cage_mass_cen_y
    x[16] = cage_mass_cen_z

    Info_es = expansion_subcall(x, Info_tc, mod_es)
    delta_b_o, delta_b_i, Info_brcs = ball_race_contact_strain(
        x, Info_es, mod_brcs
    )
    Q_b_o, Q_b_i, Info_brcf = ball_race_contact_force(
        delta_b_o, delta_b_i, mod_brcf
    )
    Info_cf = ball_centrifugal_forece(x, Info_brcs, Info_brcf, mod_cf)

    x[29:24+12*n:12] = Info_cf[0][:, 2, 0]
    x[31:24+12*n:12] = Info_cf[0][:, 0, 0]
    x[35:24+12*n:12] = Info_cf[0][:, 1, 0]
    x[19] = np.mean(x[29:24+12*n:12]) * av_ratio

    F_b_r, M_b_r, Info_brtf = ball_race_traction_force(
        x, Info_tc, Info_brcs, Info_brcf, mod_brtf
    )

    if max_rs == 2:
        delta_b_o_, delta_b_i_ = np.zeros((n, 1, 1)), np.zeros((n, 1, 1))
        Info_brcs_ = (
            np.zeros((n, 3, 3)), np.zeros((n, 3, 3)), np.zeros((n, 1, 1)),
            np.zeros((n, 1, 1)), np.zeros((n, 1, 1)), np.zeros((n, 1, 1)),
            np.zeros((n, 1, 1)), np.zeros((n, 1, 1)), 0., 0.
        )

        Q_b_o_, Q_b_i_ = np.zeros((n, 1, 1)), np.zeros((n, 1, 1))
        Info_brcf_ = (
            np.zeros((n, 1, 1)), np.zeros((n, 1, 1)), np.zeros((n, 1, 1)),
            np.zeros((n, 1, 1)), np.zeros((n, 1, 1)), np.zeros((n, 1, 1)),
            np.zeros((n, 1, 1)), np.zeros((n, 1, 1)), np.zeros((n, 1, 1)),
            np.zeros((n, 1, 1)), np.zeros((n, 1, 1)), np.zeros((n, 1, 1))
        )

        F_b_r_ = (
            np.zeros((n, 3, 1)), np.zeros((n, 3, 1)),
            np.zeros((1, 3, 1)), np.zeros((1, 3, 1))
        )
        M_b_r_ = (
            np.zeros((n, 3, 1)), np.zeros((n, 3, 1)),
            np.zeros((1, 3, 1)), np.zeros((1, 3, 1))
        )
        Info_brtf_ = (
            np.zeros((n, 1, 1)), np.zeros((n, 1, 1)), np.zeros((n, 1, 1)),
            np.zeros((n, 1, 1)), np.zeros((n, 3, 1)), np.zeros((n, 3, 1)),
            np.zeros((n, 1, 1)), np.zeros((n, 1, 1)), np.zeros((n, 1, 1)),
            np.zeros((n, 1, 1)), np.zeros((n, 3, 1)), np.zeros((n, 3, 1)),
            np.zeros((n, 3, 1)), np.zeros((n, 3, 1)), np.zeros((n, 3, 1)),
            np.zeros((n, 3, 1)), np.zeros((n, 1, 1)), np.zeros((n, 1, 1)),
            np.zeros((n, 1, 1)), np.zeros((n, 1, 1)), np.zeros((n, 1, 1)),
            np.zeros((n, 1, 1)))
    else:
        delta_b_o_, delta_b_i_, Info_brcs_ = ball_race_contact_strain_(
            Info_brcs, mod_brcs_
        )

        Q_b_o_, Q_b_i_, Info_brcf_ = ball_race_contact_force_(
            delta_b_o_, delta_b_i_, Info_brcs_, mod_brcf_
        )
        F_b_r_, M_b_r_, Info_brtf_ = ball_race_traction_force_(
            x, Info_tc, Info_brcs, Info_brcs_,
            Info_brcf_, Info_brtf, mod_brtf_
        )

    if k_life_freq > 0:
        Info_fl = fatigue_life(
            x, Info_es, Info_brcs, Info_brcs_,
            Info_brcf, Info_brcf_, Info_brtf,
            Info_brtf_, Info_flc, mod_fl
        )

    Info = (Info_es,
            # Information of expansion_subcall.
            Info_brcs,
            # Information of ball_race_contact_strain.
            Info_brcf,
            # Information of ball_race_contact_force.
            Info_brtf,
            # Information of ball_race_traction_force.
            Info_brcs_,
            # Information of ball_race_contact_strain_.
            Info_brcf_,
            # Information of ball_race_contact_force_.
            Info_brtf_,
            # Information of ball_race_traction_force_.
            Info_fl
            # Information of fatigue_life.
            )

    np.save('Initial_value.npy', x)
