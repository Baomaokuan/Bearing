# -*- coding: utf-8 -*-
"""
Created on Tue Feb 01 8:00:00 2025

@author: Baomaokuan's Chengguo
Program: Bearing Analysis of Mechanical Kinetics-b(V1.0a) Quasi_statics
"""

###############################################################################
#                                Input library                                #
###############################################################################
import math
import numba
import pickle
import numpy as np

from Input import *
from scipy import special
from Mat_properties import *
from Oil_properties import *
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
    """
    if r1 > 0:
        u0 = -r*((1 - po) + (1 + po)*(r1/r)**2)*r0**2/(E*(r0**2 - r1**2))
        u1 = r*((1 - po) + (1 + po)*(r0/r)**2)*r1**2/(E*(r0**2 - r1**2))
        u2 =                                                                  \
            (den*r*0.125*(3 + po)*(1 - po)*(r0**2 + r1**2 +
                                            (1 + po)*(r0*r1/r)**2/(1 - po) -
                                            (1 + po)*r**2/(3 + po))/E)

        st0 = -(1 + (r1/r)**2)*r0**2/(r0**2 - r1**2)
        st1 = (1 + (r0/r)**2)*r1**2/(r0**2 - r1**2)
        st2 =                                                                 \
            den*0.125*(3 + po)*(r0**2 + r1**2+(r0*r1/r)**2 -
                                (1 + 3*po)*r**2/(3 + po))
    else:
        u0 = -r*(1 - po)/E
        u1 = 0
        u2 = den*r*0.125*(1 - po)*((3 + po)*r0**2 - (1 + po)*r**2)/E

        st0 = -1
        st1 = 0
        st2 = den*0.125*(3 + po)*(r0**2 - (1 + 3*po)*r**2/(3 + po))

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
    deltah0, deltah1, deltah2, hooph0, hooph1, hooph2 =                       \
        flatdisk(denh, eh, poh, dh/2, dio/2, dio/2)
    vh00, vh01, vh02 = deltah0, deltah1, deltah2
    sth00, sth01, sth02 = hooph0, hooph1, hooph2
    vh03 = dio/2*cteh

    deltas0, deltas1, deltas2, hoops0, hoops1, hoops2 =                       \
        flatdisk(dens, es, pos, dii/2, ds/2, dii/2)
    vs00, vs01, vs02 = deltas0, deltas1, deltas2
    sts00, sts01, sts02 = hoops0, hoops1, hoops2
    vs03 = dii/2*ctes

    deltao10, deltao11, deltao12, hoopo10, hoopo11, hoopo12 =                 \
        flatdisk(deno, eo, poo, dio/2, dyo/2, dio/2)
    vo00, vo01, vo02 = deltao10, deltao11, deltao12
    sto00, sto01, sto02 = hoopo10, hoopo11, hoopo12
    vo03 = dio/2*cteo

    deltao20, deltao21, deltao22, hoopo20, hoopo21, hoopo22 =                 \
        flatdisk(deno, eo, poo, dio/2, dyo/2, dyo/2)
    vo10, vo11, vo12 = deltao20, deltao21, deltao22
    sto10, sto11, sto12 = hoopo20, hoopo21, hoopo22
    vo13 = dyo/2*cteo

    deltai10, deltai11, deltai12, hoopi10, hoopi11, hoopi12 =                 \
        flatdisk(deni, ei, poi, dyi/2, dii/2, dyi/2)
    vi00, vi01, vi02 = deltai10, deltai11, deltai12
    sti00, sti01, sti02 = hoopi10, hoopi11, hoopi12
    vi03 = dyi/2*ctei

    deltai20, deltai21, deltai22, hoopi20, hoopi21, hoopi22 =                 \
        flatdisk(deni, ei, poi, dyi/2, dii/2, dii/2)
    vi10, vi11, vi12 = deltai20, deltai21, deltai22
    sti10, sti11, sti12 = hoopi20, hoopi21, hoopi22
    vi13 = dii/2*ctei

    if nc == 0:
        vc00, vc01, vc02, vc03 = 0, 0, 0, 0
        vc10, vc11, vc12, vc13 = 0, 0, 0, 0
        stc00, stc01, stc02 = 0, 0, 0
        stc10, stc11, stc12 = 0, 0, 0
    else:
        deltac10, deltac11, deltac12, hoopc10, hoopc11, hoopc12 =             \
            flatdisk(denc, ec, poc, dco/2, dci/2, dco/2)
        vc00, vc01, vc02 = deltac10, deltac11, deltac12
        stc00, stc01, stc02 = hoopc10, hoopc11, hoopc12
        vc03 = dco/2*ctec

        deltac20, deltac21, deltac22, hoopc20, hoopc21, hoopc22 =             \
            flatdisk(denc, ec, poc, dco/2, dci/2, dci/2)
        vc10, vc11, vc12 = deltac20, deltac21, deltac22
        stc10, stc11, stc12 = hoopc20, hoopc21, hoopc22
        vc13 = dci*ctec
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
@numba.njit(fastmath = False)
def expansion_subcall(Info_tc, mod_es):
    """
        Solve the clearance change.

        Parameters
        ----------
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
    D_p_u, R_o_m, Fit_i_s, Fit_o_h, n_cseg, vh01, vh02, vh03, vs00, vs02, vs03,     \
        vo00, vo02, vo03, vo10, vo12, vo13, vi01, vi02, vi03, vi11, vi12,     \
        vi13, vc02,  vc03, vc12, vc13, rpm_i, rpm_o, sto10, sto12, sti01,     \
        sti02, sti11, sti12, stc12 = mod_es[0::]

    temp_o, temp_i, temp_h, temp_s, temp_c, temp_r = Info_tc[0:6]

    fitso, fitsi = Fit_o_h/2, Fit_i_s/2

    race_ang_vel_o, race_ang_vel_i = rpm_o*math.pi/30, rpm_i*math.pi/30
    cage_ang_vel = 0.5*(race_ang_vel_o + race_ang_vel_i)
    ###########################################################################
    #                             Geometry change                             #
    ###########################################################################
    race_hoop_o = sto12*race_ang_vel_o**2
    race_fhoop_o = race_hoop_o

    race_hoop_i = sti12*race_ang_vel_i**2
    race_fhoop_i = sti02*race_ang_vel_i**2

    race_u_o_0 = vo02*race_ang_vel_o**2 + vo03*(temp_o - temp_r)
    race_u_o_1 = vo12*race_ang_vel_o**2 + vo13*(temp_o - temp_r)

    race_u_i_0 = vi02*race_ang_vel_i**2 + vi03*(temp_i - temp_r)
    race_u_i_1 = vi12*race_ang_vel_i**2 + vi13*(temp_i - temp_r)

    hsng_u = vh02*race_ang_vel_o**2 + vh03*(temp_h - temp_r)
    op_race_fit_o = fitso + race_u_o_0 - hsng_u
    if op_race_fit_o > 0:
        P_o = op_race_fit_o/(vh01 - vo00)
        race_u_o_0 = race_u_o_0 + P_o*vo00
        race_u_o_1 = race_u_o_1 + P_o*vo10
        race_hoop_o = race_hoop_o + P_o*sto10
        race_fhoop_o = race_fhoop_o + race_hoop_o

    shft_u = vs02*race_ang_vel_i**2 + vs03*(temp_s - temp_r)
    op_race_fit_i = fitsi + shft_u - race_u_i_1
    if op_race_fit_i > 0:
        P_i = op_race_fit_i/(vi11 - vs00)
        race_u_i_0 = race_u_i_0 + P_i*vi01
        race_u_i_1 = race_u_i_1 + P_i*vi11
        race_hoop_i = race_hoop_i + P_i*sti11
        race_fhoop_i = race_fhoop_i + P_i*sti01

    op_race_fit_o, op_race_fit_i = 2*op_race_fit_o, 2*op_race_fit_i
    race_exp_o, race_exp_i = race_u_o_1, race_u_i_0

    if n_cseg == 0:
        cage_u_0, cage_u_1, cage_poc_u = 0, 0, 0
    else:    
        cage_u_0 = vc02*cage_ang_vel**2 + vc03*(temp_c - temp_r)
        cage_u_1 = vc12*cage_ang_vel**2 + vc13*(temp_c - temp_r)
        cage_poc_u = cage_u_0*D_p_u/R_o_m
        cage_hoop = stc12*cage_ang_vel**2

    tauh_o_GZ = -0.5*race_fhoop_o
    tauh_i_GZ = -0.5*race_fhoop_i
    tauh_o_IH = -0.4714045*race_fhoop_o
    tauh_i_IH = -0.4714045*race_fhoop_i
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
#                         Calculate contact stiffness                         #
###############################################################################
@numba.njit(fastmath = False)
def modified_stiffness(r0x, r0y, r1x, r1y, E0, po0, E1, po1, stype):
    """Solve the contact stiffness.

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
    E = 2/(((1 - po0**2)/E0) + ((1 - po1**2)/E1))
    e = 2/E
    rx = r0x*r1x/(r0x + r1x)
    ry = r0y*r1y/(r0y + r1y)
    r = rx*ry/(rx + ry)
    sum_ = 1/r0x + 1/r0y + 1/r1x + 1/r1y

    if stype == 1:
        fr = abs((1/r0x + 1/r1x - 1/r0y - 1/r1y)/sum_)
    else:
        fr = (1/r0x + 1/r1x - 1/r0y - 1/r1y)/sum_

    if fr <= 0:
        a_ = 1
        b_ = 1
        del_ = 1
        e1 = 0
        e2 = 0
        k = 1
        K = 2*((2/del_)**3/sum_)/1.5*E
    elif fr >= 1:
        a_= 10
        b_ = 0
        del_ = 0
        e1 = 0
        e2 = 0
        k = 1e10
        K = 2*((2/del_)**3/sum_)/1.5*E
    else:
        fris0 = fri.shape[0]
        if fr <= fri[0]:
            m0 = 0
            m1 = 1
        else:
            i = 0
            while fr > fri[i] and i < fris0 - 1:
                i = i + 1
            m1 = i
            m0 = i - 1
        ba = bai[m0] + (bai[m1] - bai[m0])*(fr - fri[m0])/(fri[m1] - fri[m0])
        e1 = ae[m0] + (ae[m1] - ae[m0])*(fr - fri[m0])/(fri[m1] - fri[m0]) + 1
        e2 = (1 + ba**2 - fr*(1 - ba**2))*e1/(2*ba**2)
        del_ = (2*e2/math.pi)*(math.pi*ba**2/(2*e1))**(1/3)
        a_ = (2*e1/(math.pi*ba**2))**(1/3)
        b_ = (2*e1*ba/math.pi)**(1/3)
        k = a_/b_
        K = math.pi/ba*E*(r*e1/(4.5*e2**3))**0.5
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    Info_s = (K,
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

    return Info_s


###############################################################################
#                         Calculate race radius change                        #
###############################################################################
@numba.njit(fastmath = False)
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
    imp: float
        Imperfections of race geometry.
    var0: float
        Deviation of the semi-major axis from nominal or radius.
    var1: float
        Deviation of the semi-minor axis from nominal or radius.
    var2: float
        Orientation (deg) of major axis relative in x direction.

    Returns
    -------
    r: np.darray
        Effective radius.
    """
    ###########################################################################
    #                           Cylindrical radius                            #
    ###########################################################################
    if imp == 0:
        r = np.zeros_like(phi)
        r[:,0,0] = r0
    elif imp == 1:
        aa = r0 + var0
        bb = r0 + var1
        r = 1/np.sqrt((np.cos(phi - var2)/aa)**2 + (np.sin(phi - var2)/bb)**2)
    elif imp == 2:
        r = r0 + var0*np.sin(var1*phi + var2)

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
    miso = np.zeros((3,3))
    miso[0,0] = math.cos(thetay0)*math.cos(thetaz0)
    miso[0,1] = math.sin(thetaz0)
    miso[0,2] = -math.sin(thetay0)*math.cos(thetaz0)
    miso[1,0] = -math.cos(thetay0)*math.sin(thetaz0)
    miso[1,1] = math.cos(thetaz0)
    miso[1,2] = math.sin(thetay0)*math.sin(thetaz0)
    miso[2,0] = math.sin(thetay0)
    miso[2,2] = math.cos(thetay0)
    ###########################################################################
    #                   Initial misalignment of inner race                    #
    ###########################################################################
    misi = np.zeros((3,3))
    misi[0,0] = math.cos(thetay1)*math.cos(thetaz1)
    misi[0,1] = math.sin(thetaz1)
    misi[0,2] = -math.sin(thetay1)*math.cos(thetaz1)
    misi[1,0] = -math.cos(thetay1)*math.sin(thetaz1)
    misi[1,1] = math.cos(thetaz1)
    misi[1,2] = math.sin(thetay1)*math.sin(thetaz1)
    misi[2,0] = math.sin(thetay1)
    misi[2,2] = math.cos(thetay1)

    return miso, misi


###############################################################################
#                         Calculate no load position                          #
###############################################################################
def no_load_position(Info_es, mod_nlp):
    """Store the results.

    Parameters
    ----------

    Returns
    -------
    x_no_load: np.darray
        No load position of component.
    Info_nlp: tuple
        Information of no_load_pos.
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    D_b, D_m, F_x, F_y, F_z, R_b, Shim_thknss_i, Shim_thknss_o, f_i, f_o,     \
        free_con_ang, k_geo_imc_type_i, k_geo_imc_type_o, mis_i_y, mis_i_z,   \
        mis_o_y, mis_o_z, n, var_i_r0, var_i_r1, var_i_r2, var_o_r0, var_o_r1,\
        var_o_r2, shim_ang_o = mod_nlp[0::]


    P_rad_o = (D_m - 2*(f_o - 0.5)*D_b*math.cos(free_con_ang))/2 + Info_es[9]
    P_rad_i = (D_m + 2*(f_i - 0.5)*D_b*math.cos(free_con_ang))/2 + Info_es[10]
    ###########################################################################
    #                     Outer race initial misalignment                     #
    ###########################################################################
    T_omis_o = np.zeros((1,3,3))
    T_omis_o[0,:,:] = np.identity(3)
    T_I_omis = np.zeros((1,3,3))
    T_I_omis[0,0,0] = math.cos(mis_o_y)*math.cos(mis_o_z)
    T_I_omis[0,0,1] = math.sin(mis_o_z)
    T_I_omis[0,0,2] = -math.sin(mis_o_y)*math.cos(mis_o_z)
    T_I_omis[0,1,0] = -math.cos(mis_o_y)*math.sin(mis_o_z)
    T_I_omis[0,1,1] = math.cos(mis_o_z)
    T_I_omis[0,1,2] = math.sin(mis_o_y)*math.sin(mis_o_z)
    T_I_omis[0,2,0] = math.sin(mis_o_y)
    T_I_omis[0,2,2] = math.cos(mis_o_y)
    ###########################################################################
    #                     Inner race initial misalignment                     #
    ###########################################################################
    T_imis_i = np.zeros((1,3,3))
    T_imis_i[0,:,:] = np.identity(3)
    T_I_imis = np.zeros((1,3,3))
    T_I_imis[0,0,0] = math.cos(mis_i_y)*math.cos(mis_i_z)
    T_I_imis[0,0,1] = math.sin(mis_i_z)
    T_I_imis[0,0,2] = -math.sin(mis_i_y)*math.cos(mis_i_z)
    T_I_imis[0,1,0] = -math.cos(mis_i_y)*math.sin(mis_i_z)
    T_I_imis[0,1,1] = math.cos(mis_i_z)
    T_I_imis[0,1,2] = math.sin(mis_i_y)*math.sin(mis_i_z)
    T_I_imis[0,2,0] = math.sin(mis_i_y)
    T_I_imis[0,2,2] = math.cos(mis_i_y)
    ###########################################################################
    #                               Ball azimuth                              #
    ###########################################################################
    dsi = 2*math.pi/n
    phi_b = np.zeros((n,1,1))
    T_I_a = np.zeros((n,3,3))
    T_I_a[:,0,0] = 1
    for i in range(0, n):
        phi_b[i,0,0] = i*dsi
        T_I_a[i,1,1] = math.cos(phi_b[i,0,0])
        T_I_a[i,1,2] = math.sin(phi_b[i,0,0])
        T_I_a[i,2,1] = -T_I_a[i,1,2]
        T_I_a[i,2,2] = T_I_a[i,1,1]
    # T_I_b = T_I_a
    # T_b_I = np.transpose(T_I_b, (0,2,1))

    theta = math.atan2(-F_y, F_z)
    t0 = np.zeros((1,3,3))
    t0[:,0,0] = 1
    t0[:,1,1] = math.cos(theta)
    t0[:,1,2] = math.sin(theta)
    t0[:,2,1] = -t0[:,1,2]
    t0[:,2,2] = t0[:,1,1]

    e_i = race_radius(
        0, P_rad_i, phi_b[0:1,:,:], k_geo_imc_type_i, var_i_r0, var_i_r1,
        var_i_r2
        )
    e_o = race_radius(
        0, P_rad_o, phi_b[0:1,:,:], k_geo_imc_type_o, var_o_r0, var_o_r1,
        var_o_r2
        )

    rc = np.zeros((1,3,1))
    rc[0,0,0] = 0.5*(Shim_thknss_i - Shim_thknss_o)
    rc[0,2,0] = e_i - e_o

    aa = f_o*D_b + f_i*D_b - D_b
    if aa < rc[0,2,0]:
        ax = -rc[0,0,0]
        ar = 0
    elif F_x != 0:
        ax = math.sqrt(aa**2 - rc[0,2,0]**2) - rc[0,0,0]
        ar = 0
    elif F_y != 0 or F_z != 0:
        ar = math.sqrt(aa**2 - rc[0,0,0]**2) - rc[0,2,0]
        ax = 0
    ###########################################################################
    #                       Set no load race positions                        #
    ###########################################################################
    r_ig_I = np.zeros((1,3,1))
    r_ig_I[0,0,0] = ax
    r_ig_I[0,1,0] = ar*t0[0,2,1]
    r_ig_I[0,2,0] = ar*t0[0,2,2]

    r_ogc_o_a = np.zeros((1,3,1))
    r_ogc_o_a[0,0,0] = -Shim_thknss_o/2
    r_ogc_o_a[0,2,0] = e_o

    r_igc_i_a = np.zeros((1,3,1))
    r_igc_i_a[0,0,0] = Shim_thknss_i/2
    r_igc_i_a[0,2,0] = e_i
    r_igc_i_a = r_igc_i_a + T_I_a@r_ig_I

    rc = r_igc_i_a - r_ogc_o_a
    alfa = math.atan(rc[0,0,0]/rc[0,2,0])
    sa = math.sin(alfa)
    ca = math.cos(alfa)
    ar = f_o*D_b - R_b
    ###########################################################################
    #                       Set no load ball positions                        #
    ###########################################################################
    r_bg_I = np.zeros((n,3,1))
    if ax == 0:
        r_bg_I[:,0,0] = 0
        r_bg_I[:,1,0] = r_ogc_o_a[:,2,0] + ar*math.cos(shim_ang_o)
        r_bg_I[:,2,0] = phi_b[:,0,0]
    else:
        r_bg_I[:,0,0] = r_ogc_o_a[:,0,0] + ar*sa
        r_bg_I[:,1,0] = r_ogc_o_a[:,2,0] + ar*ca
        r_bg_I[:,2,0] = phi_b[:,0,0]
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    x_no_load = np.zeros(37+12*n)
    x_no_load[0] = r_ig_I[0,0,0]
    x_no_load[1] = r_ig_I[0,1,0]
    x_no_load[2] = r_ig_I[0,2,0]
    x_no_load[24:24+12*n:12] = r_bg_I[:,0,0]
    x_no_load[26:24+12*n:12] = r_bg_I[:,1,0]
    x_no_load[28:24+12*n:12] = r_bg_I[:,2,0]

    Info_nlp = (P_rad_o,
                # Race pressure radius of inner race.
                P_rad_i
                # Race pressure radius of inner race.
                )

    return x_no_load, Info_nlp


###############################################################################
#                         Calculate initial position                          #
###############################################################################
def initial_pos(x_no_load, Info_nlp, mod_ip, mod_ree):
    """Solve stle lubrication factor.

    Parameters
    ----------
    x_no_load: np.darray
        No load position of component.

    Returns
    -------
    x_init: np.darray
        Initial position of component..
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    F_x, F_y, F_z, R_b, n = mod_ip[0::]
    ###########################################################################
    #                       Initial stiffness position                        #
    ###########################################################################
    x_stiff = np.zeros(37)
    x_stiff[0:36] = x_no_load[0:36]
    x_stiff[24] = x_no_load[24] + 1e-3*R_b
    ###########################################################################
    #                           Initial displayment                           #
    ###########################################################################
    if F_x != 0:
        x_stiff_new, Info_ree =                                               \
            rolling_element_equation(x_stiff, 0, 0, Info_nlp, mod_ree)
        delta_b_o, Q_b_o = Info_ree[0], Info_ree[4]
        k_b_o = Q_b_o/delta_b_o
        Q_b_o_ave = F_x/n
        re_dis_x, re_dis_r = 1e-3*R_b + Q_b_o_ave/k_b_o, 0
        race_dis_x, race_dis_r = 2*re_dis_x, 0
    elif F_y != 0 or F_z != 0:
        x_stiff[26] = x_no_load[26] + 1e-3*R_b
        x_stiff_new, Info_ree =                                               \
            rolling_element_equation(x_stiff, 0, 0, Info_nlp, mod_ree)
        delta_b_o, Q_b_o = Info_ree[0], Info_ree[4]
        delta_b_o_max = np.max(delta_b_o, keepdims = True)
        k_b_o = 1.5*Q_b_o/delta_b_o_max
        F_norm = math.sqrt(F_y**2 + F_z**2)
        re_dis_r, re_dis_x = F_norm/k_b_o, 0
        race_dis_r, race_dis_x = 2*re_dis_r, 0
    ###########################################################################
    #                       Set initial race positions                        #
    ###########################################################################
    F_norm = math.sqrt(F_y**2 + F_z**2)
    if F_norm == 0:
        cth, sth = 1, 0
    else:
        cth, sth = F_z/F_norm, -F_y/F_norm

    x_init = np.copy(x_no_load)
    x_init[0] = x_no_load[0] + race_dis_x
    x_init[2] = x_no_load[2] - race_dis_r*sth
    x_init[4] = x_no_load[4]  + race_dis_r*cth
    ###########################################################################
    #                       Set initial ball positions                        #
    ###########################################################################
    ssi = np.zeros(n)
    ssi[:,] = np.sin(x_no_load[28:24+12*n:12])

    csi = np.zeros(n)
    csi[:,] = np.cos(x_no_load[28:24+12*n:12])

    x_init[24:24+12*n:12] = x_init[24:24+12*n:12] + re_dis_x
    x_init[26:24+12*n:12] =                                                   \
        x_init[26:24+12*n:12] + re_dis_r*(csi[:,]*cth + ssi[:,]*sth)
    x_init[28:24+12*n:12] = x_no_load[28:24+12*n:12]

    return x_init


###############################################################################
#         Calculate rolling element equation and new positionequation         #
###############################################################################
@numba.njit(fastmath = False)
def rolling_element_equation(x_old, solve_angv, solve_eqn, Info_nlp, mod_ree):
    """Solve rolling element equation.

    Parameters
    ----------
    x_old: np.darray
        Solution vector.
    solve_angv: float
        Solve angular velocity control.
    solve_eqn: float
        Solve equation control.
    Info_nlp: tuple
        Information of the no_load_position.
    mod_ree: tuple
        Mode data of rolling_element_equation.

    Returns
    -------
    x_new: np.darray
        Solution vector.
    Info_es: tuple
        Information of rolling_element_equation.
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    (
         D_b, E_b_i, E_b_o, I_b_z, K_b_i, K_b_o, Pr_rad_i, Pr_rad_o, R_b,
         R_b_i, R_b_o, R_brg_orb, Shim_thknss_i, Shim_thknss_o, T_I_imis,
         T_I_omis, brg_ang_vel, brg_mov_dir, de_b_i, de_b_o, ee_b_i, ee_b_o,
         f_i, f_o, k_brg_mov, k_geo_imc_type_i, k_geo_imc_type_o, ke_b_i,
         ke_b_o, m_b, omega_i, omega_o, var_i_r0, var_i_r1, var_i_r2, var_o_r0,
         var_o_r1, var_o_r2
         ) = mod_ree[0::]

    P_rad_o, P_rad_i = Info_nlp[0::]

    X_r, Y_r, Z_r, X_theta_r, Y_theta_r, Z_theta_r = x_old[0:12:2,]
    x_b, r_b, phi_b, x_theta_b, y_theta_b, z_theta_b = x_old[24:36:2,]

    T_omis_o = np.zeros((3,3))
    T_omis_o[0,0] = 1
    T_omis_o[1,1] = 1
    T_omis_o[2,2] = 1

    T_I_o = np.dot(T_omis_o, T_I_omis)
    T_o_I = T_I_o.T

    T_imis_i = np.zeros((3,3))
    T_imis_i[0,0] = 1
    T_imis_i[1,1] = 1
    T_imis_i[2,2] = 1

    T_I_o2 = np.dot(T_imis_i, T_I_imis)
    T_i_I = T_I_o2.T

    T_I_a = np.zeros((3,3))
    T_I_a[0,0] = 1
    T_I_a[1,1] = math.cos(phi_b)
    T_I_a[1,2] = math.sin(phi_b)
    T_I_a[2,1] = -T_I_a[1,2]
    T_I_a[2,2] = T_I_a[1,1]

    r_og_I = np.zeros((3,1))
    r_og_I[0,0] = 0
    r_og_I[1,0] = 0
    r_og_I[2,0] = 0

    r_ig_I = np.zeros((3,1))
    r_ig_I[0,0] = X_r
    r_ig_I[1,0] = Y_r
    r_ig_I[2,0] = Z_r

    r_bg_I = np.zeros((3,1))
    r_bg_I[0,0] = x_b
    r_bg_I[1,0] = -r_b*math.sin(phi_b)
    r_bg_I[2,0] = r_b*math.cos(phi_b)
    ###########################################################################
    #         Transformation relationship between ball and outer race         #
    ###########################################################################
    r_bg_og_I = r_bg_I - r_og_I
    r_bg_og_o = np.dot(T_o_I, r_bg_og_I)

    phi_b_o = math.atan2(-r_bg_og_o[1,0], r_bg_og_o[2,0])

    T_o_oa = np.zeros((3,3))
    T_o_oa[0,0] = 1
    T_o_oa[1,1] = math.cos(phi_b_o)
    T_o_oa[1,2] = math.sin(phi_b_o)
    T_o_oa[2,1] = -T_o_oa[1,2]
    T_o_oa[2,2] = T_o_oa[1,1]
    T_oa_o = T_o_oa.T

    r_bg_og_oa = np.dot(T_o_oa, r_bg_og_o)

    e_o = race_radius(
        0, P_rad_o, np.array([[[phi_b_o]]]), k_geo_imc_type_o, var_o_r0,
        var_o_r1, var_o_r2
        )

    r_ogc_o = np.zeros((3,1))
    r_ogc_o[0,0] = -Shim_thknss_o/2
    r_ogc_o[1,0] = -e_o[0,0,0]*math.sin(phi_b_o)
    r_ogc_o[2,0] = e_o[0,0,0]*math.cos(phi_b_o)
    r_ogc_I = np.dot(T_I_o, r_ogc_o)
    r_ogc_oa = np.dot(T_o_oa, r_ogc_o)

    r_bg_ogc_oa = r_bg_og_oa - r_ogc_oa
    r_bg_ogc_oa_norm =                                                        \
        (r_bg_ogc_oa[0,0]**2 + r_bg_ogc_oa[1,0]**2 +
         r_bg_ogc_oa[2,0]**2)**0.5

    e_bg_ogc_oa = r_bg_ogc_oa/r_bg_ogc_oa_norm
    e_bg_ogc_o = np.dot(T_oa_o, e_bg_ogc_oa)
    e_bg_ogc_I = np.dot(T_o_I, e_bg_ogc_o)
    e_bg_ogc_a = np.dot(T_I_a, e_bg_ogc_I)

    delta_b_o = r_bg_ogc_oa_norm - (f_o - 0.5)*D_b
    if delta_b_o < 1e-10:
        delta_b_o = 0

    alpha_b_o_0 = math.atan2(e_bg_ogc_a[0,0], e_bg_ogc_a[2,0])
    if abs(e_bg_ogc_a[0,0]/e_bg_ogc_a[2,0]) < 1e-6:
        alpha_b_o_0 = 0

    alpha_b_o_1 = math.atan2(e_bg_ogc_a[0,0], e_bg_ogc_a[2,0])
    if abs(e_bg_ogc_a[1,0]) < 1e-5:
        alpha_b_o_1 = 0
    else:
        alpha_b_o_1 =                                                         \
            math.atan2(-e_bg_ogc_a[1,0],
                       math.sqrt(e_bg_ogc_a[0,0]**2 + e_bg_ogc_a[2,0]**2))

    Q_b_o = K_b_o*delta_b_o**1.5
    if Q_b_o > 0:
        dqn_b_o = Q_b_o/delta_b_o
    else:
        dqn_b_o = 0

    aa_b_o =                                                                  \
        (Q_b_o**(1/3))*(6*ke_b_o**2*de_b_o*R_b_o/(math.pi*E_b_o))**(1/3)

    bb_b_o = aa_b_o/ke_b_o

    if Q_b_o > 0:
        p_max_b_o = 1.5*Q_b_o/(math.pi*aa_b_o*bb_b_o)
    else:
        p_max_b_o = 0
    ###########################################################################
    #         Transformation relationship between ball and inner race         #
    ###########################################################################
    r_bg_ig_I = r_bg_I - r_ig_I
    r_bg_ig_i = np.dot(T_I_o2, r_bg_ig_I)

    phi_b_i = math.atan2(-r_bg_ig_i[1,0], r_bg_ig_i[2,0])

    T_i_ia = np.zeros((3,3))
    T_i_ia[0,0] = 1
    T_i_ia[1,1] = math.cos(phi_b_i)
    T_i_ia[1,2] = math.sin(phi_b_i)
    T_i_ia[2,1] = -T_i_ia[1,2]
    T_i_ia[2,2] = T_i_ia[1,1]
    T_ia_i = T_i_ia.T

    r_bg_ig_ia = np.dot(T_i_ia, r_bg_ig_i)

    e_i = race_radius(
        0, P_rad_i, np.array([[[phi_b_i]]]), k_geo_imc_type_i, var_i_r0,
        var_i_r1, var_i_r2
        )

    r_igc_i = np.zeros((3,1))
    r_igc_i[0,0] = Shim_thknss_i/2
    r_igc_i[1,0] = -e_i[0,0,0]*math.sin(phi_b_i)
    r_igc_i[2,0] = e_i[0,0,0]*math.cos(phi_b_i)
    r_igc_I = np.dot(T_i_I, r_igc_i)
    r_igc_ia = np.dot(T_i_ia, r_igc_i)

    r_bg_igc_ia = r_bg_ig_ia - r_igc_ia
    r_bg_igc_ia_norm =                                                        \
        (r_bg_igc_ia[0,0]**2 + r_bg_igc_ia[1,0]**2 +
         r_bg_igc_ia[2,0]**2)**0.5

    e_bg_igc_ia = r_bg_igc_ia/r_bg_igc_ia_norm
    e_bg_igc_i = np.dot(T_ia_i, e_bg_igc_ia)
    e_bg_igc_I = np.dot(T_i_I, e_bg_igc_i)
    e_bg_igc_a = np.dot(T_I_a, e_bg_igc_I)

    delta_b_i = r_bg_igc_ia_norm - (f_i - 0.5)*D_b
    if delta_b_i < 1e-10:
        delta_b_i = 0

    alpha_b_i_0 = math.atan2(e_bg_igc_a[0,0], e_bg_igc_a[2,0])
    if abs(e_bg_igc_a[0,0]/e_bg_igc_a[2,0]) < 1e-6:
        alpha_b_i_0 = 0

    alpha_b_i_1 = math.atan2(e_bg_igc_a[0,0], e_bg_igc_a[2,0])
    if abs(e_bg_igc_a[1,0]) < 1e-5:
        alpha_b_i_1 = 0
    else:
        alpha_b_i_1 =                                                         \
            math.atan2(-e_bg_igc_a[1,0],
                       math.sqrt(e_bg_igc_a[0,0]**2 + e_bg_igc_a[2,0]**2))

    Q_b_i = K_b_i*delta_b_i**1.5
    if Q_b_i > 0:
        dqn_b_i = Q_b_i/delta_b_i
    else:
        dqn_b_i = 0

    aa_b_i =                                                                  \
        (Q_b_i**(1/3))*(6*ke_b_i**2*de_b_i*R_b_i/(math.pi*E_b_i))**(1/3)

    bb_b_i = aa_b_i/ke_b_i

    if Q_b_i > 0:
        p_max_b_i = 1.5*Q_b_i/(math.pi*aa_b_i*bb_b_i)
    else:
        p_max_b_i = 0
    ###########################################################################
    #    Transformation relationship between ball and secondary outer race    #
    ###########################################################################
    _delta_b_o, _Q_b_o, _alpha_b_o_0, _alpha_b_o_1 = 0, 0, 0, 0
    if Shim_thknss_o > 0:
        _r_bg_og_oa = np.dot(T_o_oa, r_bg_og_o)

        _r_ogc_o = np.zeros((3,1))
        _r_ogc_o[0,0] = Shim_thknss_o/2
        _r_ogc_o[1,0] = -e_o[0,0,0]*math.sin(phi_b_o)
        _r_ogc_o[2,0] = e_o[0,0,0]*math.cos(phi_b_o)
        _r_ogc_oa = np.dot(T_o_oa, _r_ogc_o)

        _r_bg_ogc_oa = _r_bg_og_oa - _r_ogc_oa
        _r_bg_ogc_oa_norm =                                                   \
            (_r_bg_ogc_oa[0,0]**2 + _r_bg_ogc_oa[1,0]**2 +
             _r_bg_ogc_oa[2,0]**2)**0.5

        _e_bg_ogc_oa = _r_bg_ogc_oa/_r_bg_ogc_oa_norm
        _e_bg_ogc_o = np.dot(T_oa_o, _e_bg_ogc_oa)
        _e_bg_ogc_I = np.dot(T_o_I, _e_bg_ogc_o)
        _e_bg_ogc_a = np.dot(T_I_a, _e_bg_ogc_I)

        _delta_b_o = _r_bg_ogc_oa_norm - (f_o - 0.5)*D_b
        if _delta_b_o < 1e-10:
             _delta_b_o = 0

        _alpha_b_o_0 = math.atan2(_e_bg_ogc_a[0,0], _e_bg_ogc_a[2,0])
        if abs(_e_bg_ogc_a[0,0]/_e_bg_ogc_a[2,0]) < 1e-6:
            _alpha_b_o_0 = 0

        _alpha_b_o_1 = math.atan2(_e_bg_ogc_a[0,0], _e_bg_ogc_a[2,0])
        if abs(_e_bg_ogc_a[1,0]) < 1e-5:
            alpha_b_o_1 = 0
        else:
            _alpha_b_o_1 = math.atan2(
                -_e_bg_ogc_a[1,0],
                math.sqrt(_e_bg_ogc_a[0,0]**2 + _e_bg_ogc_a[2,0]**2)
                )

        _Q_b_o = K_b_o*_delta_b_o**1.5
        if _Q_b_o > 0:
            _dqn_b_o = _Q_b_o/_delta_b_o
        else:
            _dqn_b_o = 0

        _aa_b_o =                                                             \
            (_Q_b_o**(1/3))*(6*ke_b_o**2*de_b_o*
                              R_b_o/(math.pi*E_b_o))**(1/3)

        _bb_b_o = _aa_b_o/ke_b_o

        if _Q_b_o > 0:
            _p_max_b_o = 1.5*_Q_b_o/(math.pi*_aa_b_o*_bb_b_o)
        else:
            _p_max_b_o = 0
    ###########################################################################
    #    Transformation relationship between ball and secondary inner race    #
    ###########################################################################
    _delta_b_i, _Q_b_i, _alpha_b_i_0, _alpha_b_i_1 = 0, 0, 0, 0
    if Shim_thknss_i > 0:
        _r_bg_ig_ia = np.dot(T_i_ia, r_bg_ig_i)

        _r_igc_i = np.zeros((3,1))
        _r_igc_i[0,0] = -Shim_thknss_i/2
        _r_igc_i[1,0] = -e_i[0,0,0]*math.sin(phi_b_i)
        _r_igc_i[2,0] = e_i[0,0,0]*math.cos(phi_b_i)
        _r_igc_ia = np.dot(T_i_ia, _r_igc_i)

        _r_bg_igc_ia = _r_bg_ig_ia - _r_igc_ia
        _r_bg_igc_ia_norm =                                                   \
            (_r_bg_igc_ia[0,0]**2 + _r_bg_igc_ia[1,0]**2 +
             _r_bg_igc_ia[2,0]**2)**0.5

        _e_bg_igc_ia = _r_bg_igc_ia/_r_bg_igc_ia_norm
        _e_bg_igc_i = np.dot(T_ia_i, _e_bg_igc_ia)
        _e_bg_igc_I = np.dot(T_i_I, _e_bg_igc_i)
        _e_bg_igc_a = np.dot(T_I_a, _e_bg_igc_I)

        _delta_b_i = _r_bg_igc_ia_norm - (f_i - 0.5)*D_b
        if _delta_b_i < 1e-10:
             _delta_b_i = 0

        _alpha_b_i_0 = math.atan2(_e_bg_igc_a[0,0], _e_bg_igc_a[2,0])
        if abs(_e_bg_igc_a[0,0]/_e_bg_igc_a[2,0]) < 1e-6:
            _alpha_b_i_0 = 0

        _alpha_b_i_1 = math.atan2(_e_bg_igc_a[0,0], _e_bg_igc_a[2,0])
        if abs(_e_bg_igc_a[1,0]) < 1e-5:
            alpha_b_i_1 = 0
        else:
            _alpha_b_i_1 = math.atan2(
                -_e_bg_igc_a[1,0],
                math.sqrt(_e_bg_igc_a[0,0]**2 + _e_bg_igc_a[2,0]**2)
                )

        _Q_b_i = K_b_i*_delta_b_i**1.5
        if _Q_b_i > 0:
            _dqn_b_i = _Q_b_i/_delta_b_i
        else:
            _dqn_b_i = 0

        _aa_b_i =                                                             \
            (_Q_b_i**(1/3))*(
                6*ke_b_i**2*de_b_i*R_b_i/(math.pi*E_b_i))**(1/3)

        _bb_b_i = _aa_b_i/ke_b_i

        if _Q_b_i > 0:
            _p_max_b_i = 1.5*_Q_b_i/(math.pi*_aa_b_i*_bb_b_i)
        else:
            _p_max_b_i = 0
    ###########################################################################
    #                    No need to solve angular velocity                    #
    ###########################################################################
    if solve_angv == 0:
        C = np.zeros((3,1))
        C[0,0] = x_old[31,]
        C[1,0] = x_old[35,]
        C[2,0] = x_old[29,]

        F_c = m_b*x_old[26,]*C[2,0]**2
        G_m = -I_b_z*C[1,0]*C[2,0]
    ###########################################################################
    #                     Need to solve angular velocity                      #
    ###########################################################################
    else:
        T_a_oc = np.zeros((3,3))
        T_a_oc[0,0] = math.cos(alpha_b_o_0)
        # T_a_oc[0,1] = 0
        T_a_oc[0,2] = -math.sin(alpha_b_o_0)
        T_a_oc[1,0] = math.sin(alpha_b_o_0)*math.sin(alpha_b_o_1)
        T_a_oc[1,1] = math.cos(alpha_b_o_1)
        T_a_oc[1,2] = math.cos(alpha_b_o_0)*math.sin(alpha_b_o_1)
        T_a_oc[2,0] = math.sin(alpha_b_o_0)*math.cos(alpha_b_o_1)
        T_a_oc[2,1] = -math.sin(alpha_b_o_1)
        T_a_oc[2,2] = math.cos(alpha_b_o_0)*math.cos(alpha_b_o_1)

        T_I_oc = np.dot(T_a_oc, T_I_a)

        r_os_bg_oc = np.zeros((3,1))
        # r_os_bg_oc[0,0] = 0
        # r_os_bg_oc[1,0] = 0
        r_os_bg_oc[2,0] = R_b

        r_bg_og_oc = np.dot(T_I_oc, r_bg_og_I)
        r_os_og_oc = r_os_bg_oc + r_bg_og_oc

        T_a_ic = np.zeros((3,3))
        T_a_ic[0,0] = math.cos(alpha_b_i_0)
        # T_a_ic[0,1] = 0
        T_a_ic[0,2] = -math.sin(alpha_b_i_0)
        T_a_ic[1,0] = math.sin(alpha_b_i_0)*math.sin(alpha_b_i_1)
        T_a_ic[1,1] = math.cos(alpha_b_i_1)
        T_a_ic[1,2] = math.cos(alpha_b_i_0)*math.sin(alpha_b_i_1)
        T_a_ic[2,0] = math.sin(alpha_b_i_0)*math.cos(alpha_b_i_1)
        T_a_ic[2,1] = -math.sin(alpha_b_i_1)
        T_a_ic[2,2] = math.cos(alpha_b_i_0)*math.cos(alpha_b_i_1)

        T_I_o2c = np.dot(T_a_ic, T_I_a)

        r_is_bg_ic = np.zeros((3,1))
        # r_is_bg_ic[0,0] = 0
        # r_is_bg_ic[1,0] = 0
        r_is_bg_ic[2,0] = R_b

        r_bg_ig_ic = np.dot(T_I_o2c, r_bg_ig_I)
        r_is_ig_ic = r_is_bg_ic + r_bg_ig_ic

        cond_0 =                                                              \
            Q_b_o*aa_b_o*ee_b_o*math.cos(math.pi + alpha_b_i_0 - alpha_b_o_0)
        cond_1 = Q_b_i*aa_b_i*ee_b_i

        abs_alpha_b_i_0 = math.pi + alpha_b_i_0

        if cond_0 > cond_1:
            aI = alpha_b_o_0
            aJ = abs_alpha_b_i_0
            oI = omega_o
            oJ = omega_i
            c3I = (Pr_rad_o**2 - (1*0.34729636*aa_b_o)**2)**0.5 - Pr_rad_o
            cb1I = r_os_bg_oc[0,0]
            cb1J = r_is_bg_ic[0,0]
            cr1I = r_os_og_oc[0,0]
            cr1J = r_is_ig_ic[0,0]
            cb3I = r_os_bg_oc[2,0]
            cb3J = r_is_bg_ic[2,0]
            cr3I = r_os_og_oc[2,0]
            cr3J = r_is_ig_ic[2,0]
        else:
            aI = abs_alpha_b_i_0
            aJ = alpha_b_o_0
            oI = omega_i
            oJ = omega_o
            c3I = (Pr_rad_i**2 - (0*0.34729636*aa_b_i)**2)**0.5 - Pr_rad_i
            cb1I = r_is_bg_ic[0,0]
            cb1J = r_os_bg_oc[0,0]
            cr1I = r_is_ig_ic[0,0]
            cr1J = r_os_og_oc[0,0]
            cb3I = r_is_bg_ic[2,0]
            cb3J = r_os_bg_oc[2,0]
            cr3I = r_is_ig_ic[2,0]
            cr3J = r_os_og_oc[2,0]

        X =                                                                   \
            (cb1I*math.sin(aI) - cr1I*math.sin(aI) +
             (cr3I + c3I)*math.cos(aI))
        Y = -cr1J*math.sin(aJ) + cr3J*math.cos(aJ)

        A = np.zeros((3,3))
        A[0,0] = math.sin(aI)
        A[0,1] = math.cos(aI)
        A[0,2] = math.sin(aI)
        A[1,0] = (cb3I + c3I)*math.cos(aI)
        A[1,1] = -(cb3I + c3I)*math.sin(aI)
        A[1,2] = X
        A[2,0] = -cb1J*math.sin(aJ) + cb3J*math.cos(aJ)
        A[2,1] = -cb1J*math.cos(aJ) - cb3J*math.sin(aJ)
        A[2,2] = Y

        B = np.zeros((3,1))
        B[0,0] = oI*math.sin(aI)
        B[1,0] = oI*X
        B[2,0] = oJ*Y

        C = np.linalg.solve(A, B)

        F_c = m_b*x_old[26,]*C[2,0]**2
        G_m = -I_b_z*C[1,0]*C[2,0]

        if cond_0 > cond_1:
            ic = 0
        else:
            ic = 1
        ixlm = 1 - ic/2
        fg0 = ixlm*R_b*G_m
        fg1 = (1 - ixlm)*R_b*G_m
    ###########################################################################
    #                 No need update rolling element position                 #
    ###########################################################################
    if solve_eqn == 0:
        x_new = np.copy(x_old)
        g = np.zeros((2,1))
    ###########################################################################
    #                     Update rolling element position                     #
    ###########################################################################
    else:
        sin_alpha_b_o_0 = math.sin(alpha_b_o_0)
        cos_alpha_b_o_0 = math.cos(alpha_b_o_0)

        # sin_alpha_b_o_1 = math.sin(alpha_b_o_1)
        # cos_alpha_b_o_1 = math.cos(alpha_b_o_1)

        sin_alpha_b_i_0 = math.sin(alpha_b_i_0)
        cos_alpha_b_i_0 = math.cos(alpha_b_i_0)

        # sin_alpha_b_i_1 = math.sin(alpha_b_i_1)
        # cos_alpha_b_i_1 = math.cos(alpha_b_i_1)

        _sin_alpha_b_o_0 = math.sin(_alpha_b_o_0)
        _cos_alpha_b_o_0 = math.cos(_alpha_b_o_0)

        # _sin_alpha_b_o_1 = math.sin(_alpha_b_o_1)
        # _cos_alpha_b_o_1 = math.cos(_alpha_b_o_1)

        _sin_alpha_b_i_0 = math.sin(_alpha_b_i_0)
        _cos_alpha_b_i_0 = math.cos(_alpha_b_i_0)

        # _sin_alpha_b_i_1 = math.sin(_alpha_b_i_1)
        # _cos_alpha_b_i_1 = math.cos(_alpha_b_i_1)

        if k_brg_mov > 0:
            if brg_mov_dir == 0:
                F_b_mov_x_I = 0
                F_b_mov_y_I =                                                 \
                    (2*brg_ang_vel*r_bg_I[1,0]*C[2,0] +
                     r_bg_I[1,0]*brg_ang_vel**2)*m_b
                F_b_mov_z_I =                                                 \
                    (2*brg_ang_vel*r_bg_I[2,0]*C[2,0] +
                     (r_bg_I[2,0] + R_brg_orb)*brg_ang_vel**2)*m_b
                F_b_mov_x_a = F_b_mov_x_I
                F_b_mov_r_a = (-F_b_mov_y_I*math.sin(x_old[28]) +
                               F_b_mov_z_I*math.cos(x_old[28]))
            elif brg_mov_dir == 1:
                F_b_mov_x_I =                                                 \
                    (2*brg_ang_vel*r_bg_I[1,0]*C[2,0] +
                     (r_bg_I[0,0] + R_brg_orb)*brg_ang_vel**2)*m_b
                F_b_mov_y_I =                                                 \
                    (0 +
                     r_bg_I[2,0]*brg_ang_vel**2)*m_b
                F_b_mov_x_a = F_b_mov_x_I
                F_b_mov_r_a = F_b_mov_y_I*math.cos(x_old[28])
            elif brg_mov_dir == 2:
                F_b_mov_x_I =                                                 \
                    (2*brg_ang_vel*r_bg_I[2,0]*C[2,0] +
                     (r_bg_I[0,0] + R_brg_orb)*brg_ang_vel**2)*m_b
                F_b_mov_y_I =                                                 \
                    (0 +
                     r_bg_I[1,0]*brg_ang_vel**2)*m_b
                F_b_mov_x_a = F_b_mov_x_I
                F_b_mov_r_a = F_b_mov_y_I*math.cos(x_old[28])
            else:
                F_b_mov_x_a, F_b_mov_r_a = 0, 0
        else:
            F_b_mov_x_a, F_b_mov_r_a = 0, 0

        g = np.zeros((2,1))
        g[0,0] = -fg0*cos_alpha_b_o_0 - fg1*cos_alpha_b_i_0 - F_b_mov_x_a
        g[1,0] = -F_c + fg0*sin_alpha_b_o_0 + fg1*sin_alpha_b_i_0 - F_b_mov_r_a
        g[0,0] = g[0,0] + Q_b_o*sin_alpha_b_o_0 + Q_b_i*sin_alpha_b_i_0
        g[1,0] = g[1,0] + Q_b_o*cos_alpha_b_o_0 + Q_b_i*cos_alpha_b_i_0
        g[0,0] = g[0,0] + _Q_b_o*_sin_alpha_b_o_0 + _Q_b_i*_sin_alpha_b_i_0
        g[1,0] = g[1,0] + _Q_b_o*_cos_alpha_b_o_0 + _Q_b_i*_cos_alpha_b_i_0

        ssis = math.sin(phi_b)
        csis = math.cos(phi_b)

        r1o = r_bg_og_I - r_ogc_I
        rao = np.dot(T_I_a, r1o)

        aao = (rao[0,0]**2 + rao[1,0]**2 + rao[2,0]**2)**0.5

        ax = np.zeros((3,1))
        ax[0,0] = T_I_a[0,0]
        ax[1,0] = T_I_a[1,0]
        ax[2,0] = T_I_a[2,0]

        ar = np.zeros((3,1))
        ar[0,0] = -T_I_a[0,1]*ssis + T_I_a[0,2]*csis
        ar[1,0] = -T_I_a[1,1]*ssis + T_I_a[1,2]*csis
        ar[2,0] = -T_I_a[2,1]*ssis + T_I_a[2,2]*csis

        delxo = (rao[0,0]*ax[0,0] + rao[1,0]*ax[1,0] + rao[2,0]*ax[2,0])/aao
        delro = (rao[0,0]*ar[0,0] + rao[1,0]*ar[1,0] + rao[2,0]*ar[2,0])/aao

        dQ1dx = dqn_b_o*delxo
        dQ1dr = dqn_b_o*delro

        r1i = r_bg_ig_I - r_igc_I
        rai = np.dot(T_I_a, r1i)

        aai = (rai[0,0]**2 + rai[1,0]**2 + rai[2,0]**2)**0.5

        delxi = (rai[0,0]*ax[0,0] + rai[1,0]*ax[1,0] + rai[2,0]*ax[2,0])/aai
        delri = (rai[0,0]*ar[0,0] + rai[1,0]*ar[1,0] + rai[2,0]*ar[2,0])/aai

        dQ2dx = dqn_b_i*delxi
        dQ2dr = dqn_b_i*delri

        aa1o = ((rao[0,0]*rao[0,0] + rao[2,0]*rao[2,0])**0.5)**3

        saDBo1 = rao[2,0]*(rao[2,0]*ax[0,0] - rao[0,0]*ax[2,0])/aa1o
        saDBo2 = rao[2,0]*(rao[2,0]*ar[0,0] - rao[0,0]*ar[2,0])/aa1o
        caDBo1 = rao[0,0]*(rao[0,0]*ax[2,0] - rao[2,0]*ax[0,0])/aa1o
        caDBo2 = rao[0,0]*(rao[0,0]*ar[2,0] - rao[2,0]*ar[0,0])/aa1o

        aa1i = ((rai[0,0]*rai[0,0] + rai[2,0]*rai[2,0])**0.5)**3

        saDBi1 = rai[2,0]*(rai[2,0]*ax[0,0] - rai[0,0]*ax[2,0])/aa1i
        saDBi2 = rai[2,0]*(rai[2,0]*ar[0,0] - rai[0,0]*ar[2,0])/aa1i
        caDBi1 = rai[0,0]*(rai[0,0]*ax[2,0] - rai[2,0]*ax[0,0])/aa1i
        caDBi2 = rai[0,0]*(rai[0,0]*ar[2,0] - rai[2,0]*ar[0,0])/aa1i

        if Shim_thknss_o> 0:
            _r_ogc_I = np.dot(T_I_o, _r_ogc_o)
            _r1o = r_bg_og_I - _r_ogc_I

            _rao = np.dot(T_I_a, _r1o)

            _aao = (_rao[0,0]**2 + _rao[1,0]**2 + _rao[2,0]**2)**0.5

            _delxo =                                                          \
                (_rao[0,0]*ax[0,0] + _rao[1,0]*ax[1,0] +
                 _rao[2,0]*ax[2,0])/_aao
            _delro =                                                          \
                (_rao[0,0]*ar[0,0] + _rao[1,0]*ar[1,0] +
                 _rao[2,0]*ar[2,0])/_aao

            _dQ1dx, _dQ1dr = _dqn_b_o*_delxo, _dqn_b_o*_delro

            _aa1o = ((_rao[0,0]*_rao[0,0] + _rao[2,0]*_rao[2,0])**0.5)**3

            _saDBo1 = _rao[2,0]*(_rao[2,0]*ax[0,0] - _rao[0,0]*ax[2,0])/_aa1o
            _saDBo2 = _rao[2,0]*(_rao[2,0]*ar[0,0] - _rao[0,0]*ar[2,0])/_aa1o
            _caDBo1 = _rao[0,0]*(_rao[0,0]*ax[2,0] - _rao[2,0]*ax[0,0])/_aa1o
            _caDBo2 = _rao[0,0]*(_rao[0,0]*ar[2,0] - _rao[2,0]*ar[0,0])/_aa1o
        else:
            _dQ1dx, _dQ1dr, _saDBo1, _saDBo2, _caDBo1, _caDBo2 =              \
                0, 0, 0, 0, 0, 0

        if Shim_thknss_i> 0:
            _r_igc_I = np.dot(T_I_o2, _r_igc_i)
            _r1i = r_bg_ig_I - _r_igc_I

            _rai = np.dot(T_I_a, _r1i)

            _aai = (_rai[0,0]**2 + _rai[1,0]**2 + _rai[2,0]**2)**0.5

            _delxi =                                                          \
                (_rai[0,0]*ax[0,0] + _rai[1,0]*ax[1,0] +
                 _rai[2,0]*ax[2,0])/_aai
            _delri =                                                          \
                (_rai[0,0]*ar[0,0] + _rai[1,0]*ar[1,0] +
                 _rai[2,0]*ar[2,0])/_aai

            _dQ2dx, _dQ2dr = _dqn_b_i*_delxi, _dqn_b_i*_delri

            _aa1i = ((_rai[0,0]*_rai[0,0] + _rai[2,0]*_rai[2,0])**0.5)**3

            _saDBi1 = _rai[2,0]*(_rai[2,0]*ax[0,0] - _rai[0,0]*ax[2,0])/_aa1i
            _saDBi2 = _rai[2,0]*(_rai[2,0]*ar[0,0] - _rai[0,0]*ar[2,0])/_aa1i
            _caDBi1 = _rai[0,0]*(_rai[0,0]*ax[2,0] - _rai[2,0]*ax[0,0])/_aa1i
            _caDBi2 = _rai[0,0]*(_rai[0,0]*ar[2,0] - _rai[2,0]*ar[0,0])/_aa1i
        else:
            _dQ2dx, _dQ2dr, _saDBi1, _saDBi2, _caDBi1, _caDBi2 =              \
                0, 0, 0, 0, 0, 0

        jac0 = np.zeros((2,2))
        jac0[0,0] =                                                           \
            (dQ1dx*sin_alpha_b_o_0 + Q_b_o*saDBo1 + dQ2dx*sin_alpha_b_i_0 +
             Q_b_i*saDBi1)
        jac0[0,1] =                                                           \
            (dQ1dr*sin_alpha_b_o_0 + Q_b_o*saDBo2 + dQ2dr*sin_alpha_b_i_0 +
             Q_b_i*saDBi2)
        jac0[1,0] =                                                           \
            (dQ1dx*cos_alpha_b_o_0 + Q_b_o*caDBo1 + dQ2dx*cos_alpha_b_i_0 +
             Q_b_i*caDBi1)
        jac0[1,1] =                                                           \
            (dQ1dr*cos_alpha_b_o_0 + Q_b_o*caDBo2 + dQ2dr*cos_alpha_b_i_0 +
             Q_b_i*caDBi2) - m_b*C[2,0]**2
        jac0[0,0] = jac0[0,0] - fg0*caDBo1 - fg1*caDBi1
        jac0[0,1] = jac0[0,1] - fg0*caDBo2 - fg1*caDBi2
        jac0[1,0] = jac0[1,0] + fg0*saDBo1 + fg1*saDBi1
        jac0[1,1] = jac0[1,1] + fg0*saDBo2 + fg1*saDBi2

        _jac0 = np.zeros((2,2))
        _jac0[0,0] =                                                          \
            (_dQ1dx*_sin_alpha_b_o_0 + _Q_b_o*_saDBo1 +
             _dQ2dx*_sin_alpha_b_i_0 + _Q_b_i*_saDBi1)
        _jac0[0,1] =                                                          \
            (_dQ1dr*_sin_alpha_b_o_0 + _Q_b_o*_saDBo2 +
             _dQ2dr*_sin_alpha_b_i_0 + _Q_b_i*_saDBi2)
        _jac0[1,0] =                                                          \
            (_dQ1dx*_cos_alpha_b_o_0 + _Q_b_o*_caDBo1 +
             _dQ2dx*_cos_alpha_b_i_0 + _Q_b_i*_caDBi1)
        _jac0[1,1] =                                                          \
            (_dQ1dr*_cos_alpha_b_o_0 + _Q_b_o*_caDBo2 +
             _dQ2dr*_cos_alpha_b_i_0 + _Q_b_i*_caDBi2)

        jac = jac0 + _jac0
        if abs(jac[0,0]) < 1e-6:
            jac[0,0] = 1
        if abs(jac[1,1]) < 1e-6:
            jac[1,1] = 1

        xx_new = np.linalg.solve(jac, g)

        x_new = np.zeros_like(x_old)
        x_new[0,] = x_old[0,]
        x_new[2,] = x_old[2,]
        x_new[4,] = x_old[4,]
        x_new[24,] = x_old[24,] - xx_new[0,0]
        x_new[26,] = x_old[26,] - xx_new[1,0]
        x_new[28,] = x_old[28,]
        x_new[29,] = C[2,0]
        x_new[31,] = C[0,0]
        x_new[35,] = C[1,0]
    ###########################################################################
    #                                Store result                             #
    ###########################################################################
    Info_ree = (delta_b_o,
                # Contact deflection between ball and outer race.
                 delta_b_i,
                # Contact deflection between ball and inner race.
                _delta_b_o,
                # Contact deflection between ball and secondary outer race.
                _delta_b_i,
                # Contact deflection between ball and secondary inner race.
                Q_b_o,
                # Contact force between ball and outer race.
                Q_b_i,
                # Contact force between ball and inner race.
                _Q_b_o,
                # Contact force between ball and secondary outer race.
                _Q_b_i,
                # Contact force between ball and secondary inner race.
                alpha_b_o_0,
                # Contact angle between ball and outer race.
                alpha_b_i_0,
                # Contact angle between ball and inner race.
                _alpha_b_o_0,
                # Contact angle between ball and secondary outer race.
                _alpha_b_i_0,
                # Contact angle between ball and secondary inner race.
                g
                # Equation vector.
                )

    return x_new, Info_ree


###############################################################################
#              Calculate race equation and new positionequation               #
###############################################################################
@numba.njit(fastmath = False)
def race_equation(x_old, Info_nlp, mod_re):
    """Solve race equation.

    Parameters
    ----------
    x_old: np.darray
        Solution vector.
    Info_nlp: tuple
        Information of the no_load_position.
    mod_re: tuple
        Mode data of race_equation.

    Returns
    -------
    x_new: np.darray
        Solution vector.
    Info_es: tuple
        Information of race_equation.
    """
    ###########################################################################
    #                                 Prepare                                 #
    ###########################################################################
    (
        D_b, F_r_mov, F_x, F_y, F_z, K_b_i, Shim_thknss_i, T_I_imis,
        brg_mov_dir, f_i, k_brg_mov, k_geo_imc_type_i, n, var_i_r0, var_i_r1,
        var_i_r2
        ) = mod_re[0::]

    P_rad_i = Info_nlp[1]

    X_r, Y_r, Z_r = x_old[0:6:2]
    Y_theta_r, Z_theta_r = x_old[8:12:2]
    x_b_array, r_b_array, phi_b_array =                                       \
        x_old[24:24+12*n:12], x_old[26:24+12*n:12], x_old[28:24+12*n:12]

    T_imis_i = np.zeros((3,3))
    T_imis_i[0,0] = 1
    T_imis_i[1,1] = 1
    T_imis_i[2,2] = 1

    T_I_o2 = np.dot(T_imis_i, T_I_imis)
    T_i_I = T_I_o2.T

    T_I_a_arr = np.zeros((n,3,3))
    T_I_a_arr[:,0,0] = 1
    T_I_a_arr[:,1,1] = np.cos(phi_b_array[:])
    T_I_a_arr[:,1,2] = np.sin(phi_b_array[:])
    T_I_a_arr[:,2,1] = -T_I_a_arr[:,1,2]
    T_I_a_arr[:,2,2] = T_I_a_arr[:,1,1]

    r_ig_I = np.zeros((3,1))
    r_ig_I[0,0] = X_r
    r_ig_I[1,0] = Y_r
    r_ig_I[2,0] = Z_r

    r_bg_I_arr = np.zeros((n,3,1))
    r_bg_I_arr[:,0,0] = x_b_array[:]
    r_bg_I_arr[:,1,0] = -r_b_array[:]*np.sin(phi_b_array[:])
    r_bg_I_arr[:,2,0] = r_b_array[:]*np.cos(phi_b_array[:])

    r_bg_ig_I_arr = r_bg_I_arr - r_ig_I
    ###########################################################################
    #           Transformation relationship between ball and  race            #
    ###########################################################################
    (
        r_igc_I_arr, _r_igc_I_arr, r_bg_igc_a_arr_norm,
        _r_bg_igc_a_arr_norm, e_bg_igc_a_arr, _e_bg_igc_a_arr
        ) = (
            np.zeros((n,3,1)), np.zeros((n,3,1)), np.zeros((n,1,1)),
            np.zeros((n,1,1)), np.zeros((n,3,1)), np.zeros((n,3,1))
            )
    for i in range(0, n):
        T_I_a = np.zeros((3,3))
        T_I_a[0,0] = 1
        T_I_a[1,1] = T_I_a_arr[i,1,1]
        T_I_a[1,2] = T_I_a_arr[i,1,2]
        T_I_a[2,1] = T_I_a_arr[i,2,1]
        T_I_a[2,2] = T_I_a_arr[i,2,2]

        r_bg_I = np.zeros((3,1))
        r_bg_I[0,0] = r_bg_I_arr[i,0,0]
        r_bg_I[1,0] = r_bg_I_arr[i,1,0]
        r_bg_I[2,0] = r_bg_I_arr[i,2,0]
        #######################################################################
        #       Transformation relationship between ball and inner race       #
        #######################################################################
        r_bg_ig_I = r_bg_I - r_ig_I
        r_bg_ig_i = np.dot(T_I_o2, r_bg_ig_I)

        phi_b_i = math.atan2(-r_bg_ig_i[1,0], r_bg_ig_i[2,0])

        T_i_ia = np.zeros((3,3))
        T_i_ia[0,0] = 1
        T_i_ia[1,1] = math.cos(phi_b_i)
        T_i_ia[1,2] = math.sin(phi_b_i)
        T_i_ia[2,1] = -T_i_ia[1,2]
        T_i_ia[2,2] = T_i_ia[1,1]
        T_ia_i = T_i_ia.T

        r_bg_ig_ia = np.dot(T_i_ia, r_bg_ig_i)

        e_i =  race_radius(
            0, P_rad_i, np.array([[[phi_b_i]]]), k_geo_imc_type_i, var_i_r0,
            var_i_r1, var_i_r2
            )

        r_igc_i = np.zeros((3,1))
        r_igc_i[0,0] = Shim_thknss_i/2
        r_igc_i[1,0] = -e_i[0,0,0]*math.sin(phi_b_i)
        r_igc_i[2,0] = e_i[0,0,0]*math.cos(phi_b_i)
        r_igc_ia = np.dot(T_i_ia, r_igc_i)
        r_igc_I = np.dot(T_i_I, r_igc_i)

        r_bg_igc_ia = r_bg_ig_ia - r_igc_ia
        r_bg_igc_ia_norm =                                                    \
            math.sqrt(r_bg_igc_ia[0,0]**2 + r_bg_igc_ia[1,0]**2 +
                      r_bg_igc_ia[2,0]**2)

        e_bg_igc_ia = r_bg_igc_ia/r_bg_igc_ia_norm
        e_bg_igc_i = np.dot(T_ia_i, e_bg_igc_ia)
        e_bg_igc_I = np.dot(T_i_I, e_bg_igc_i)
        e_bg_igc_a = np.dot(T_I_a, e_bg_igc_I)
        #######################################################################
        #  Transformation relationship between ball and secondary inner race  #
        #######################################################################
        if Shim_thknss_i > 0:
            _r_bg_ig_ia = np.dot(T_i_ia, r_bg_ig_i)

            _r_igc_i = np.zeros((3,1))
            _r_igc_i[0,0] = -r_igc_i[0,0]
            _r_igc_i[1,0] = r_igc_i[1,0]
            _r_igc_i[2,0] = r_igc_i[2,0]
            _r_igc_ia = np.dot(T_i_ia, _r_igc_i)
            _r_igc_I = np.dot(T_i_I, _r_igc_i)

            _r_bg_igc_ia = _r_bg_ig_ia - _r_igc_ia
            _r_bg_igc_ia_norm =                                               \
                math.sqrt(_r_bg_igc_ia[0,0]**2 + _r_bg_igc_ia[1,0]**2 +
                          _r_bg_igc_ia[2,0]**2)

            _e_bg_igc_ia = _r_bg_igc_ia/_r_bg_igc_ia_norm
            _e_bg_igc_i = np.dot(T_ia_i, _e_bg_igc_ia)
            _e_bg_igc_I = np.dot(T_i_I, _e_bg_igc_i)
            _e_bg_igc_a = np.dot(T_I_a, _e_bg_igc_I)

            _r_igc_I_arr[i,:,:], _r_bg_igc_a_arr_norm[i,:,:],                 \
                _e_bg_igc_a_arr[i,:,:] =                                      \
                _r_igc_I, _r_bg_igc_ia_norm, _e_bg_igc_a

        r_igc_I_arr[i,:,:], r_bg_igc_a_arr_norm[i,:,:],                       \
            e_bg_igc_a_arr[i,:,:] =                                           \
            r_igc_I, r_bg_igc_ia_norm, e_bg_igc_a

    delta_b_i_arr = r_bg_igc_a_arr_norm - (f_i - 0.5)*D_b
    for i in range(0, n):
        if delta_b_i_arr[i,0,0] < 1e-10:
            delta_b_i_arr[i,0,0] = 0

    alpha_b_i_0_arr = np.zeros((n,1,1))
    alpha_b_i_0_arr[:,0,0] =                                                  \
        np.arctan2(e_bg_igc_a_arr[:,0,0], e_bg_igc_a_arr[:,2,0])
    for i in range(0, n):
        if e_bg_igc_a_arr[i,0,0]/e_bg_igc_a_arr[i,2,0] < 1e-6:
            alpha_b_i_0_arr[i,0,0] = -math.pi

    Q_b_i_arr = K_b_i*delta_b_i_arr**1.5

    cond_0 = np.where(delta_b_i_arr > 0)

    dqn_b_i_arr = np.zeros_like(Q_b_i_arr)
    dqn_b_i_arr[cond_0[0],0,0] =                                              \
        Q_b_i_arr[cond_0[0],0,0]/delta_b_i_arr[cond_0[0],0,0]

    if Shim_thknss_i > 0:
        _delta_b_i_arr = _r_bg_igc_a_arr_norm - (f_i - 0.5)*D_b
        for i in range(0, n):
            if _delta_b_i_arr[i,0,0] < 1e-10:
                _delta_b_i_arr[i,0,0] = 0

        _alpha_b_i_0_arr = np.zeros((n,1,1))
        _alpha_b_i_0_arr[:,0,0] =                                             \
            np.arctan2(_e_bg_igc_a_arr[:,0,0], _e_bg_igc_a_arr[:,2,0])
        for i in range(0, n):
            if _e_bg_igc_a_arr[i,0,0]/_e_bg_igc_a_arr[i,2,0] < 1e-6:
                _alpha_b_i_0_arr[i,0,0] = -math.pi

        _Q_b_i_arr = K_b_i*_delta_b_i_arr**1.5

        cond_1 = np.where(_delta_b_i_arr > 0)

        _dqn_b_i_arr = np.zeros_like(_Q_b_i_arr)
        _dqn_b_i_arr[cond_1[0],0,0] =                                         \
            _Q_b_i_arr[cond_1[0],0,0]/_delta_b_i_arr[cond_1[0],0,0]

    sa_b_i = np.zeros((n,1,1))
    sa_b_i[:,0,0] = np.sin(alpha_b_i_0_arr[:,0,0])

    ca_b_i = np.zeros((n,1,1))
    ca_b_i[:,0,0] = np.cos(alpha_b_i_0_arr[:,0,0])

    if Shim_thknss_i > 0:
        _sa_b_i = np.zeros((n,1,1))
        _sa_b_i[:,0,0] = np.sin(_alpha_b_i_0_arr[:,0,0])

        _ca_b_i = np.zeros((n,1,1))
        _ca_b_i[:,0,0] = np.cos(_alpha_b_i_0_arr[:,0,0])

    fg1 = 0

    qa = np.zeros((n,1,1))
    qa[:,0,0] = Q_b_i_arr[:,0,0]*sa_b_i[:,0,0] - fg1*ca_b_i[:,0,0]

    qr = np.zeros((n,1,1))
    qr[:,0,0] = Q_b_i_arr[:,0,0]*ca_b_i[:,0,0] - fg1*sa_b_i[:,0,0]

    _qa, _qr = np.zeros((n,1,1)), np.zeros((n,1,1))
    if Shim_thknss_i > 0:
        _qa[:,0,0] = _Q_b_i_arr[:,0,0]*_sa_b_i[:,0,0] - fg1*_ca_b_i[:,0,0]
        _qr[:,0,0] = _Q_b_i_arr[:,0,0]*_ca_b_i[:,0,0] - fg1*_sa_b_i[:,0,0]

    if k_brg_mov > 0:
        if brg_mov_dir == 0:
            k0, k1, k2 = 0, 0, 1
        elif brg_mov_dir == 1:
            k0, k1, k2 = 0, 1, 0
        elif brg_mov_dir == 2:
            k0, k1, k2 = 1, 0, 0
        else:
            k0, k1, k2 = 0, 0, 0
    else:
        k0, k1, k2 = 0, 0, 0

    G = np.zeros((3,1))
    G[0,0] = np.sum(qa[:,0,0] + _qa[:,0,0]) + F_x + k0*F_r_mov
    G[1,0] = np.sum(-(qr[:,0,0] + _qr[:,0,0])*np.sin(phi_b_array[:])
                    ) + F_y - k1*F_r_mov
    G[2,0] = np.sum((qr[:,0,0] + _qr[:,0,0]
                     )*np.cos(phi_b_array[:])) + F_z + k2*F_r_mov

    r1 = r_bg_ig_I_arr - r_igc_I_arr

    ra, ra0, ra1 = np.zeros((n,3,1)), np.zeros((3,3)), np.zeros((3,1))
    for i in range(0, n):
        ra0[0,0] = 1
        ra0[1,1] = T_I_a_arr[i,1,1]
        ra0[1,2] = T_I_a_arr[i,1,2]
        ra0[2,1] = T_I_a_arr[i,2,1]
        ra0[2,2] = T_I_a_arr[i,2,2]
        ra1[0,0] = r1[i,0,0]
        ra1[1,0] = r1[i,1,0]
        ra1[2,0] = r1[i,2,0]
        ra2 = np.dot(ra0, ra1)
        ra[i,:,:] = ra2

    ax = np.zeros((n,3,1))
    ax[:,0,0], ax[:,1,0], ax[:,2,0] =                                         \
        -T_I_a_arr[:,0,0], -T_I_a_arr[:,1,0], -T_I_a_arr[:,2,0]

    ay = np.zeros((n,3,1))
    ay[:,0,0], ay[:,1,0], ay[:,2,0] =                                         \
        -T_I_a_arr[:,0,1], -T_I_a_arr[:,1,1], -T_I_a_arr[:,2,1]

    az = np.zeros((n,3,1))
    az[:,0,0], az[:,1,0], az[:,2,0] =                                         \
        -T_I_a_arr[:,0,2], -T_I_a_arr[:,1,2], -T_I_a_arr[:,2,2]

    aa = np.zeros((n,1,1))
    aa[:,0,0] = np.sqrt(ra[:,0,0]**2 + ra[:,1,0]**2 + ra[:,2,0]**2)

    delxx = np.zeros((n,1,1))
    delxx[:,0,0] =                                                            \
        (ra[:,0,0]*ax[:,0,0] + ra[:,1,0]*ax[:,1,0] +
         ra[:,2,0]*ax[:,2,0])/aa[:,0,0]

    delyy = np.zeros((n,1,1))
    delyy[:,0,0] =                                                            \
        (ra[:,0,0]*ay[:,0,0] + ra[:,1,0]*ay[:,1,0] +
         ra[:,2,0]*ay[:,2,0])/aa[:,0,0]

    delzz= np.zeros((n,1,1))
    delzz[:,0,0] =                                                            \
        (ra[:,0,0]*az[:,0,0] + ra[:,1,0]*az[:,1,0] +
         ra[:,2,0]*az[:,2,0])/aa[:,0,0]

    con_load_dr = np.zeros((n,3,1))
    con_load_dr[:,0,0] = dqn_b_i_arr[:,0,0]*delxx[:,0,0]
    con_load_dr[:,1,0] = dqn_b_i_arr[:,0,0]*delyy[:,0,0]
    con_load_dr[:,2,0] = dqn_b_i_arr[:,0,0]*delzz[:,0,0]

    aa1 = np.zeros((n,1,1))
    aa1[:,0,0] = np.sqrt(ra[:,0,0]**2 + ra[:,2,0]**2)**3

    sa_dr = np.zeros((n,3,1))
    sa_dr[:,0,0] =                                                            \
        ra[:,2,0]*(ra[:,2,0]*ax[:,0,0] - ra[:,0,0]*ax[:,2,0])/aa1[:,0,0]
    sa_dr[:,1,0] =                                                            \
        ra[:,2,0]*(ra[:,2,0]*ay[:,0,0] - ra[:,0,0]*ay[:,2,0])/aa1[:,0,0]
    sa_dr[:,2,0] =                                                            \
        ra[:,2,0]*(ra[:,2,0]*az[:,0,0] - ra[:,0,0]*az[:,2,0])/aa1[:,0,0]

    ca_dr = np.zeros((n,3,1))
    ca_dr[:,0,0] =                                                            \
        ra[:,0,0]*(ra[:,0,0]*ax[:,2,0] - ra[:,2,0]*ax[:,0,0])/aa1[:,0,0]
    ca_dr[:,1,0] =                                                            \
        ra[:,0,0]*(ra[:,0,0]*ay[:,2,0] - ra[:,2,0]*ay[:,0,0])/aa1[:,0,0]
    ca_dr[:,2,0] =                                                            \
        ra[:,0,0]*(ra[:,0,0]*az[:,2,0] - ra[:,2,0]*az[:,0,0])/aa1[:,0,0]

    if Shim_thknss_i > 0:
        _r1 = np.zeros((n,3,1))
        _r1[:,0,0] = r_bg_ig_I_arr[:,0,0] - _r_igc_I_arr[:,0,0]
        _r1[:,1,0] = r_bg_ig_I_arr[:,1,0] - _r_igc_I_arr[:,1,0]
        _r1[:,2,0] = r_bg_ig_I_arr[:,2,0] - _r_igc_I_arr[:,2,0]

        _ra, _ra0, _ra1 = np.zeros((n,3,1)), np.zeros((3,3)), np.zeros((3,1))
        for i in range(0, n):
            _ra0[0,0] = 1
            _ra0[1,1] = T_I_a_arr[i,1,1]
            _ra0[1,2] = T_I_a_arr[i,1,2]
            _ra0[2,1] = T_I_a_arr[i,2,1]
            _ra0[2,2] = T_I_a_arr[i,2,2]
            _ra1[0,0] = _r1[i,0,0]
            _ra1[1,0] = _r1[i,1,0]
            _ra1[2,0] = _r1[i,2,0]
            _ra2 = np.dot(_ra0, _ra1)
            _ra[i,:,:] = _ra2

        _aa = np.zeros((n,1,1))
        _aa[:,0,0] = np.sqrt(_ra[:,0,0]**2 + _ra[:,1,0]**2 + _ra[:,2,0]**2)

        _delxx = np.zeros((n,1,1))
        _delxx[:,0,0] =                                                       \
            (_ra[:,0,0]*ax[:,0,0] + _ra[:,1,0]*ax[:,1,0] +
             _ra[:,2,0]*ax[:,2,0])/_aa[:,0,0]

        _delyy = np.zeros((n,1,1))
        _delyy[:,0,0] =                                                       \
            (_ra[:,0,0]*ay[:,0,0] + _ra[:,1,0]*ay[:,1,0] +
             _ra[:,2,0]*ay[:,2,0])/_aa[:,0,0]

        _delzz= np.zeros((n,1,1))
        _delzz[:,0,0] =                                                       \
            (_ra[:,0,0]*az[:,0,0] + _ra[:,1,0]*az[:,1,0] +
             _ra[:,2,0]*az[:,2,0])/_aa[:,0,0]

        _con_load_dr = np.zeros((n,3,1))
        _con_load_dr[:,0,0] = _dqn_b_i_arr[:,0,0]*_delxx[:,0,0]
        _con_load_dr[:,1,0] = _dqn_b_i_arr[:,0,0]*_delyy[:,0,0]
        _con_load_dr[:,2,0] = _dqn_b_i_arr[:,0,0]*_delzz[:,0,0]

        _aa1 = np.zeros((n,1,1))
        _aa1[:,0,0] = np.sqrt(_ra[:,0,0]**2 + _ra[:,2,0]**2)**3

        _sa_dr = np.zeros((n,3,1))
        _sa_dr[:,0,0] =                                                       \
            _ra[:,2,0]*(_ra[:,2,0]*ax[:,0,0] -
                        _ra[:,0,0]*ax[:,2,0])/_aa1[:,0,0]
        _sa_dr[:,1,0] =                                                       \
            _ra[:,2,0]*(_ra[:,2,0]*ay[:,0,0] -
                        _ra[:,0,0]*ay[:,2,0])/_aa1[:,0,0]
        _sa_dr[:,2,0] =                                                       \
            _ra[:,2,0]*(_ra[:,2,0]*az[:,0,0] -
                        _ra[:,0,0]*az[:,2,0])/_aa1[:,0,0]

        _ca_dr = np.zeros((n,3,1))
        _ca_dr[:,0,0] =                                                       \
            _ra[:,0,0]*(_ra[:,0,0]*ax[:,2,0] -
                        _ra[:,2,0]*ax[:,0,0])/_aa1[:,0,0]
        _ca_dr[:,1,0] =                                                       \
            _ra[:,0,0]*(_ra[:,0,0]*ay[:,2,0] -
                        _ra[:,2,0]*ay[:,0,0])/_aa1[:,0,0]
        _ca_dr[:,2,0] =                                                       \
            _ra[:,0,0]*(_ra[:,0,0]*az[:,2,0] -
                        _ra[:,2,0]*az[:,0,0])/_aa1[:,0,0]

    qa_dr = con_load_dr*sa_b_i + Q_b_i_arr*sa_dr - fg1*ca_dr
    qr_dr = con_load_dr*ca_b_i + Q_b_i_arr*ca_dr + fg1*ca_dr

    _qa_dr, _qr_dr = np.zeros((n,3,1)), np.zeros((n,3,1))
    if Shim_thknss_i > 0:
        _qa_dr = _con_load_dr*_sa_b_i + _Q_b_i_arr*_sa_dr - fg1*_ca_dr
        _qr_dr = _con_load_dr*_ca_b_i + _Q_b_i_arr*_ca_dr + fg1*_ca_dr

    Jac = np.zeros((3,3))
    Jac[0,0] = np.sum(qa_dr[:,0,0] + _qa_dr[:,0,0])
    Jac[0,1] = np.sum(qa_dr[:,1,0] + _qa_dr[:,1,0])
    Jac[0,2] = np.sum(qa_dr[:,2,0] + _qa_dr[:,2,0])
    Jac[1,0] = -np.sum((qr_dr[:,0,0] + _qr_dr[:,0,0])*np.sin(phi_b_array[:]))
    Jac[1,1] = -np.sum((qr_dr[:,1,0] + _qr_dr[:,1,0])*np.sin(phi_b_array[:]))
    Jac[1,2] = -np.sum((qr_dr[:,2,0] + _qr_dr[:,2,0])*np.sin(phi_b_array[:]))
    Jac[2,0] = np.sum((qr_dr[:,0,0] + _qr_dr[:,0,0])*np.cos(phi_b_array[:]))
    Jac[2,1] = np.sum((qr_dr[:,1,0] + _qr_dr[:,1,0])*np.cos(phi_b_array[:]))
    Jac[2,2] = np.sum((qr_dr[:,2,0] + _qr_dr[:,2,0])*np.cos(phi_b_array[:]))

    xx_new = np.linalg.solve(Jac, G)

    x_new = np.zeros_like(x_old)
    x_new[0,] = x_old[0,] - xx_new[0,0]
    x_new[2,] = x_old[2,] - xx_new[1,0]
    x_new[4,] = x_old[4,] - xx_new[2,0]
    x_new[24::] = x_old[24::]
    ###########################################################################
    #                                Store result                             #
    ###########################################################################
    Info_re = (G
               # Equation vector.
               )

    return x_new, Info_re


###############################################################################
#                                Main function                                #
###############################################################################
if __name__=="__main__":
    mod_tc, mod_es, mod_nlp, mod_ip, mod_ree, mod_re =                        \
        (), (), (), (), (), ()
    mod_name = [
        'mod_tc',
        'mod_es',
        'mod_nlp',
        'mod_ip',
        'mod_ree',
        'mod_re'
        ]
    mod_index_name = [
        'mod_tc_index',
        'mod_es_index',
        'mod_nlp_index',
        'mod_ip_index',
        'mod_ree_index',
        'mod_re_index'
        ]
    mod_tc_index = (
        'ini_temp_o',
        'ini_temp_i',
        'ini_temp_h',
        'ini_temp_s',
        'ini_temp_c',
        'ini_temp_r',
        'ini_temp_chrn'
        )
    mod_es_index = (
        'D_p_u',
        'R_o_m',
        'Fit_i_s',
        'Fit_o_h',
        'n_cseg',
        'vh01',
        'vh02',
        'vh03',
        'vs00',
        'vs02',
        'vs03',
        'vo00',
        'vo02',
        'vo03',
        'vo10',
        'vo12',
        'vo13',
        'vi01',
        'vi02',
        'vi03',
        'vi11',
        'vi12',
        'vi13',
        'vc02',
        'vc03',
        'vc12',
        'vc13',
        'rpm_i',
        'rpm_o',
        'sto10',
        'sto12',
        'sti01',
        'sti02',
        'sti11',
        'sti12',
        'stc12'
        )
    mod_nlp_index = (
        'D_b',
        'D_m',
        'F_x',
        'F_y',
        'F_z',
        'R_b',
        'Shim_thknss_i',
        'Shim_thknss_o',
        'f_i',
        'f_o',
        'free_con_ang',
        'k_geo_imc_type_i',
        'k_geo_imc_type_o',
        'mis_i_y',
        'mis_i_z',
        'mis_o_y',
        'mis_o_z',
        'n',
        'var_i_r0',
        'var_i_r1',
        'var_i_r2',
        'var_o_r0',
        'var_o_r1',
        'var_o_r2',
        'shim_ang_o'
        )
    mod_ip_index = (
        'F_x',
        'F_y',
        'F_z',
        'R_b',
        'n'
        )
    mod_ree_index = (
        'D_b',
        'E_b_i',
        'E_b_o',
        'I_b_z',
        'K_b_i',
        'K_b_o',
        'Pr_rad_i',
        'Pr_rad_o',
        'R_b',
        'R_b_i',
        'R_b_o',
        'R_brg_orb',
        'Shim_thknss_i',
        'Shim_thknss_o',
        'T_I_imis',
        'T_I_omis',
        'brg_ang_vel',
        'brg_mov_dir',
        'de_b_i',
        'de_b_o',
        'ee_b_i',
        'ee_b_o',
        'f_i',
        'f_o',
        'k_brg_mov',
        'k_geo_imc_type_i',
        'k_geo_imc_type_o',
        'ke_b_i',
        'ke_b_o',
        'm_b',
        'omega_i',
        'omega_o',
        'var_i_r0',
        'var_i_r1',
        'var_i_r2',
        'var_o_r0',
        'var_o_r1',
        'var_o_r2'
        )
    mod_re_index = (
        'D_b',
        'F_r_mov',
        'F_x',
        'F_y',
        'F_z',
        'K_b_i',
        'Shim_thknss_i',
        'T_I_imis',
        'brg_mov_dir',
        'f_i',
        'k_brg_mov',
        'k_geo_imc_type_i',
        'n',
        'var_i_r0',
        'var_i_r1',
        'var_i_r2'
        )
    mod_index_data = [
        mod_name,
        mod_index_name,
        mod_tc_index,
        mod_es_index,
        mod_nlp_index,
        mod_ip_index,
        mod_ree_index,
        mod_re_index
        ]
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
            locals()[list_base_data[j]] = base_data[i][list_base_data[j]]
    ###########################################################################
    #                      Solve preparation parameters                       #
    ###########################################################################
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

    Pr_rad_o, Pr_rad_i = f_o*D_b/(2*f_o + 1), f_i*D_b/(2*f_i + 1)

    omega_o, omega_i = rpm_o*math.pi/30, rpm_i*math.pi/30
    ###########################################################################
    #                         Solve contact stiffness                         #
    ###########################################################################
    r0xo, r0yo = (D_m/math.cos(free_con_ang) + D_b)/2, f_o*D_b
    K_b_o, ke_b_o, de_b_o, t_b_o, E_b_o, ep_b_o, R_yipu_b_o, R_b_o,  a_b_o,   \
        b_b_o =  modified_stiffness(
            -r0xo, -r0yo, R_b, R_b, E_o, po_o, E_b, po_b, 0
            )
    ee_b_o  = special.ellipe(1/ke_b_o)

    r0xi, r0yi = (D_m/math.cos(free_con_ang) - D_b)/2, f_i*D_b
    K_b_i, ke_b_i, de_b_i, t_b_i, E_b_i, ep_b_i, R_yipu_b_i, R_b_i,  a_b_i,   \
        b_b_i = modified_stiffness(
            r0xi, -r0yi, R_b, R_b, E_i, po_i, E_b, po_b, 0
            )
    ee_b_i  = special.ellipe(1/ke_b_i)
    ###########################################################################
    #                       Solve initial misalignment                        #
    ###########################################################################
    T_I_omis, T_I_imis = misalignment(mis_o_y, mis_o_z, mis_i_y, mis_i_z)
    ###########################################################################
    #                             Initial seting                              #
    ###########################################################################
    F_r_mov =                                                                 \
        (m_i*brg_load_frac_i - m_o*brg_load_frac_o)*brg_ang_vel**2*R_brg_orb
    ###########################################################################
    #                          Module in the subroutine                       #
    ###########################################################################
    mod_list_name = mod_index_data[0]
    for i in range(2, len(mod_list_name) + 2):
        locals()[mod_list_name[i-2]] = []
        for j in range(0, len(mod_index_data[i])):
            if mod_index_data[i][j] in globals():
                locals()[mod_list_name[i-2]].append(
                    locals()[mod_index_data[i][j]]
                    )
        locals()[mod_list_name[i-2]] = tuple(locals()[mod_list_name[i-2]])
    ###########################################################################
    #                             Initial seting                              #
    ###########################################################################
    Info_tc = mod_tc
    Info_es = expansion_subcall(Info_tc, mod_es)

    x_no_load, Info_nlp = no_load_position(Info_es, mod_nlp)
    x_qusai = initial_pos(x_no_load, Info_nlp, mod_ip, mod_ree)

    x_old_re = np.zeros(36)
    error = np.zeros(2*n+3)
    for i in range(0, 100):
        if i == 0:
            x = np.copy(x_qusai)
        for j in range(0, 101):
            if j == 0:
                x_old_r = np.copy(x)
            x_new_r, Info_re = race_equation(x_old_r, Info_nlp, mod_re)
            error[0:3] = Info_re[:,0]
            if np.all(np.abs(error[0:3])) < 1e-8 or j >= 100:
                x[0:24] = x_new_r[0:24]
                break
            else:
                x_old_r = x_new_r

        for k in range(0, n):
            for l in range(0, 101):
                if l == 0:
                    x_old_re[0:24] = x[0:24]
                    x_old_re[24:36] = x[24+12*k:36+12*k]
                x_new_re, Info_ree =                                          \
                        rolling_element_equation(
                            x_old_re, 1, 1, Info_nlp, mod_ree
                            )
                error[2*k+3:2*k+5] = Info_ree[-1][:,0]
                if np.all(np.abs(error[2*k+3:2*k+5])) < 1e-8 or l >= 100:
                    x[24+12*k:36+12*k] = x_new_re[24:36]
                    break
                else:
                    x_old_re = x_new_re
        x_check, Info_re_check = race_equation(x, Info_nlp, mod_re)
        error[0:3] = Info_re_check[:,0]
        if np.all(np.abs(error[0:3])) < 1e-8:
            break
    x[7] = rpm_i*math.pi/30
    x[12] = np.average(x[24:24+12*n:12]) + cage_mass_cen_x
    x[14] = cage_mass_cen_y
    x[16] = cage_mass_cen_z
    x[15,] = -np.mean(x[29:24+12*n:12])*wv_ratio*x[16]
    x[17,] = np.mean(x[29:24+12*n:12])*wv_ratio*x[14]
    x[19,] = np.mean(x[29:24+12*n:12])*av_ratio
    x[31+12*n] = rpm_o*math.pi/30

    np.save('Initial_value.npy', x)