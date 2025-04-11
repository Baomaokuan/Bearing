# -*- coding: utf-8 -*-
"""
Created on Tue Feb 01 8:00:00 2025

@author: Baomaokuan's Chengguo
Program: Bearing Analysis of Mechanical Kinetics-b(V1.0a) Mat_properties
"""

###############################################################################
#                                Input Libraray                               #
###############################################################################
import math
import numba
import numpy as np
###############################################################################
#                               Tool function                                 #
###############################################################################
@numba.njit(fastmath=False)
def interpolate_1d(xi, x, y):
    """Interpolate 1D data.

    Parameters
    ----------
    xi: float
        x coordinate of desired interpolated value.
    x: array
        Independent variable array x.
    y: array
        Independent variable array y.

    Returns
    -------
    yi: float
        y coordinate of desired interpolated value.
    """
    xs0 = x.shape[0]
    if xi <= x[0]:
        m0, m1 = 0, 1
    else:
        i = 0
        while xi > x[i] and i < xs0 - 1:
            i = i + 1
        m0, m1 = i - 1, i
        
    yi = y[m0] + (y[m1] - y[m0]) * (xi - x[m0]) / (x[m1] - x[m0])

    return yi

###############################################################################
#                                Main function                                #
###############################################################################
@numba.njit(fastmath=False)
def mat_main(mat_type, curr_temp):
    """Choose mat.

    Parameters
    ----------
    oil_type: float
        Oil type.
    curr_temp: float
        Current temperature.
    curr_pres: float
        Current pressure.

    Returns
    -------
    oil_prop: tuple
        Oil properties.
    """
    if mat_type == 0:
        mat_prop = aisi_52100(curr_temp)
    elif mat_type == 1:
        mat_prop = aisi_4340(curr_temp)
    elif mat_type == 2:
        mat_prop = latrobe_m50(curr_temp)
    elif mat_type == 3:
        mat_prop = latrobe_m50nil(curr_temp)
    elif mat_type == 4:
        mat_prop = latrobe_m50vimvar(curr_temp)
    elif mat_type == 5:
        mat_prop = si3n4(curr_temp)
    elif mat_type == 6:
        mat_prop = zro2(curr_temp)
    elif mat_type == 7:
        mat_prop = al2o3(curr_temp)
    elif mat_type == 8:
        mat_prop = titanium(curr_temp)
    elif mat_type == 9:
        mat_prop = peek(curr_temp)
    elif mat_type == 10:
        mat_prop = polyamide_nylon(curr_temp)

    return mat_prop

###############################################################################
#                           Properties of material                            #
###############################################################################
@numba.njit(fastmath=False)
def aisi_52100(curr_temp):
    """Mat properties.

    Parameters
    ----------
    curr_temp: float
        Current temperature.
    curr_pres: float
        Current pressure.

    Returns
    -------
    mat_prop: tuple
        Mat properties.
    """
    mat_name = 'AISI 52100 Steel'
    tx = np.array([295., 473., 673., 873., 1073., 1273.])
    # Temp array for density,heat capicity & ther cond, the values are in K
    et = np.array([201.33, 178.58, 162.72, 103.42, 86.87, 66.88])
    # Values are in x e9Pa
    pot = np.array([0.277, 0.269, 0.255, 0.342, 0.396, 0.490])
    eslt = np.array([0.70, 1.09, 1.09, 0.30, 0.30, 0.20])
    txx = np.array([298., 477., 571., 977., 1077.])
    ctex = np.array([11.5, 12.6, 13.7, 14.9, 15.3])
    den_mat = 7827.
    # Denisty at current temp (SI)
    tcd_mat = 43.
    # Thermal conductivity at current temp (SI)
    sph_mat = 458.
    # Specific heat at current temp (SI)
    wc_mat = 5e-6
    # Wear coefficient at current temp (SI)
    hc_mat = 62.
    # Hardness at current temp (SI)
    vms_mat = 6.84e8
    # Von-Mises stress at current temp (SI)

    temp = curr_temp
    if temp <= tx[0,]:
        em_mat = et[0,] * 1e9
        # Elastic modulus at current temp (SI)
        po_mat = pot[0,]
        # Possion ratio at current temp (SI)
        elst_mat = eslt[0,] * 0.01
        # Elastic strain limit at current temp (SI)
    elif temp >= tx[-1,]:
        em_mat = et[-1,] * 1e9
        po_mat = pot[-1,]
        elst_mat = eslt[-1,] * 0.01
    else:
        em_mat = interpolate_1d(temp, tx, et) * 1e9
        po_mat = interpolate_1d(temp, tx, pot)
        elst_mat = interpolate_1d(temp, tx, eslt) * 0.01
    if temp <= txx[0]:
        cte_mat = ctex[0] * 1e-6
        # Coefficient of ther exp at current temp (SI)
    elif temp >= txx[-1]:
        cte_mat = ctex[-1] * 1e-6
    else:
        cte_mat = interpolate_1d(temp, txx, ctex) * 1e-6

    life_mat = 1
    # STLE material code.
    life_proc = 2
    # STLE processing code.
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    mat_prop = (den_mat,
                # Denisty at current temperature of material.
                em_mat,
                # Elastic modulus at current temperature of material.
                po_mat,
                # Possion ratio at current temperature of material.
                elst_mat,
                # Elastic strain limit at current temperature of material.
                cte_mat,
                # Coefficient of ther exp at current temperature of material.
                tcd_mat,
                # Thermal conductivity at current temperature of material.
                sph_mat,
                # Specific heat at current temperature of material.
                wc_mat,
                # Wear coefficient at current temperature of material.
                hc_mat,
                # Hardness at current temperature of material.
                vms_mat,
                # Von-Mises stress at current temperature of material.
                life_mat,
                # STLE material code of material.
                life_proc
                # STLE processing code of material.
                )

    return mat_prop

@numba.njit(fastmath=False)
def aisi_4340(curr_temp):
    """Mat properties.

    Parameters
    ----------
    curr_temp: float
        Current temperature.
    curr_pres: float
        Current pressure.

    Returns
    -------
    mat_prop: tuple
        Mat properties.
    """
    mat_name = 'AISI 4340 Steel'
    den_mat = 7850.
    # Denisty at current temp (SI)
    em_mat = 2.05e11
    # Elastic modulus at current temp (SI)
    po_mat = 0.29
    # Possion ratio at current temp (SI)
    elst_mat = 0.002
    # Elastic strain limit at current temp (SI)
    tcd_mat = 44.5
    # Thermal conductivity at current temp (SI)
    sph_mat = 475.
    # Specific heat at current temp (SI)
    wc_mat = 5e-5
    # Wear coefficient at current temp (SI)
    hc_mat = 36.
    # Hardness at current temp (SI)
    vms_mat = 5.9e8
    # Von-Mises stress at current temp (SI)
    tx = np.array([293., 523., 773.])
    # Temp array for density,heat capicity & ther cond, the values are in K
    ctex = np.array([12.7, 13.7, 14.5])

    temp = curr_temp
    if temp <= tx[0]:
        cte_mat = ctex[0] * 1e-6
        # coefficient of ther exp at current temp (SI)
    elif temp >= tx[-1]:
        cte_mat = ctex[-1] * 1e-6
    else:
        cte_mat = interpolate_1d(temp, tx, ctex) * 1e-6

    life_mat = None
    # STLE material code
    life_proc = None
    # STLE processing code
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    mat_prop = (den_mat,
                # Denisty at current temperature of material.
                em_mat,
                # Elastic modulus at current temperature of material.
                po_mat,
                # Possion ratio at current temperature of material.
                elst_mat,
                # Elastic strain limit at current temperature of material.
                cte_mat,
                # Coefficient of ther exp at current temperature of material.
                tcd_mat,
                # Thermal conductivity at current temperature of material.
                sph_mat,
                # Specific heat at current temperature of material.
                wc_mat,
                # Wear coefficient at current temperature of material.
                hc_mat,
                # Hardness at current temperature of material.
                vms_mat,
                # Von-Mises stress at current temperature of material.
                life_mat,
                # STLE material code of material.
                life_proc
                # STLE processing code of material.
                )

    return mat_prop

@numba.njit(fastmath=False)
def latrobe_m50(curr_temp):
    """Mat properties.

    Parameters
    ----------
    curr_temp: float
        Current temperature.
    curr_pres: float
        Current pressure.

    Returns
    -------
    mat_prop: tuple
        Mat properties.
    """
    mat_name = 'Latrobe CM-50 ASTM M-50 Steel'
    den_mat = 7830.
    # Denisty at current temp (SI)
    em_mat = 2.03e11
    # Elastic modulus at current temp (SI)
    po_mat = 0.28
    # Possion ratio at current temp (SI)
    elst_mat = 0.002
    # Elastic strain limit at current temp (SI)
    tcd_mat = 13.4
    # Thermal conductivity at current temp (SI)
    sph_mat = 460.
    # Specific heat at current temp (SI)
    wc_mat = 5e-6
    # Wear coefficient at current temp (SI)
    hc_mat = 64.5
    # Hardness at current temp (SI)
    vms_mat = 7.17e8
    # Von-Mises stress at current temp (SI)

    temp = curr_temp
    if temp <= 477.:
        cte_mat = 1.13e-5
        # coefficient of ther exp at current temp (SI)
    else:
        cte_mat = 1.298e-5

    life_mat = 6
    # STLE material code
    life_proc = 2
    # STLE processing code
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    mat_prop = (den_mat,
                # Denisty at current temperature of material.
                em_mat,
                # Elastic modulus at current temperature of material.
                po_mat,
                # Possion ratio at current temperature of material.
                elst_mat,
                # Elastic strain limit at current temperature of material.
                cte_mat,
                # Coefficient of ther exp at current temperature of material.
                tcd_mat,
                # Thermal conductivity at current temperature of material.
                sph_mat,
                # Specific heat at current temperature of material.
                wc_mat,
                # Wear coefficient at current temperature of material.
                hc_mat,
                # Hardness at current temperature of material.
                vms_mat,
                # Von-Mises stress at current temperature of material.
                life_mat,
                # STLE material code of material.
                life_proc
                # STLE processing code of material.
                )

    return mat_prop

@numba.njit(fastmath=False)
def latrobe_m50nil(curr_temp):
    """Mat properties.

    Parameters
    ----------
    curr_temp: float
        Current temperature.
    curr_pres: float
        Current pressure.

    Returns
    -------
    mat_prop: tuple
        Mat properties.
    """
    mat_name = 'CBS-50 NiL VIMVAR'
    den_mat = 7850.
    # Denisty at current temp (SI)
    po_mat = 0.28
    # Possion ratio at current temp (SI)
    tx = np.array([293.0, 477.0, 589.0, 700.0])
    # Temp array for density,heat capicity & ther cond, the values are in K
    ctex = np.array([11.0, 11.2, 11.6, 12.0])
    txx = np.array([295.55, 311.11, 366.66, 422.22, 477.77, 533.33, 588.88,
                    644.44, 700.0, 755.55, 811.11])
    hcx = np.array([61.8, 62.1, 61.3, 60.9, 60.7, 60.6, 60.0 ,59.1, 58.8,
                    57.7, 56.7])

    temp = curr_temp
    if temp <= tx[0]:
        cte_mat = ctex[0] * 1e-6
        # coefficient of ther exp at current temp (SI)
    elif temp >= tx[-1]:
        cte_mat = ctex[-1] * 1e-6
    else:
        cte_mat = interpolate_1d(temp, tx, ctex) * 1e-6
    em_mat = 2.208e11
    # Elastic modulus at current temp (SI)
    if temp > 294.:
        em_mat = em_mat * (1 + -9.1752313e-4 * (temp - 294.))
    if temp <= txx[0]:
        hc_mat = hcx[0]
        # Hardness at current temp (SI)
    elif temp >= txx[-1]:
        hc_mat = hcx[-1]
    else:
        hc_mat = interpolate_1d(temp, txx, hcx)
    tcd_mat = 13.4
    # Thermal conductivity at current temp (SI)
    sph_mat = 460.
    # Specific heat at current temp (SI)
    elst_mat = 0.002
    # Elastic strain limit at current temp (SI)
    wc_mat = 5e-6
    # Wear coefficient at current temp (SI)
    vms_mat = 5.79e8
    # Von-Mises stress at current temp (SI)
    vms_mat = vms_mat * em_mat / 2.034e11

    life_mat = 41
    # STLE material code
    life_proc = 7
    # STLE processing code
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    mat_prop = (den_mat,
                # Denisty at current temperature of material.
                em_mat,
                # Elastic modulus at current temperature of material.
                po_mat,
                # Possion ratio at current temperature of material.
                elst_mat,
                # Elastic strain limit at current temperature of material.
                cte_mat,
                # Coefficient of ther exp at current temperature of material.
                tcd_mat,
                # Thermal conductivity at current temperature of material.
                sph_mat,
                # Specific heat at current temperature of material.
                wc_mat,
                # Wear coefficient at current temperature of material.
                hc_mat,
                # Hardness at current temperature of material.
                vms_mat,
                # Von-Mises stress at current temperature of material.
                life_mat,
                # STLE material code of material.
                life_proc
                # STLE processing code of material.
                )

    return mat_prop

@numba.njit(fastmath=False)
def latrobe_m50vimvar(curr_temp):
    """Mat properties.

    Parameters
    ----------
    curr_temp: float
        Current temperature.
    curr_pres: float
        Current pressure.

    Returns
    -------
    mat_prop: tuple
        Mat properties.
    """
    mat_name = 'Lescalloy M-50 VIMVAR Bearing Steel'
    den_mat = 8027.172
    # Denisty at current temp (SI)
    po_mat = 0.28
    # Possion ratio at current temp (SI)
    tcd_mat = 13.4
    # Thermal conductivity at current temp (SI)
    sph_mat = 460.
    # Specific heat at current temp (SI)
    elst_mat = 0.002
    # Elastic strain limit at current temp (SI)
    wc_mat = 5e-6
    # Wear coefficient at current temp (SI)
    vms_mat = 7.17e8
    # Von-Mises stress at current temp (SI)
    em_mat = 2.034e11
    # Elastic modulus at current temp (SI)

    temp = curr_temp
    if temp > 294.:
        em_mat  = em_mat * (1 + -9.1752313 * (temp - curr_temp))
    vms_mat = vms_mat * em_mat / 2.034e11
    tx = np.array([294.44, 366.66, 477.77, 588.88, 700.0, 755.55, 811.11,
                   866.66, 922.22])
    # Temp array for density,heat capicity & ther cond, the values are in K
    hcx = np.array([62.5, 62.0, 60.5, 59.0, 56.5, 55.0, 53.0, 47.5, 30.0])
    txx = np.array([294.0, 366.0, 422.0, 477.0, 533.0, 589.0, 644.0, 700.0,
                    755.0, 811.0])
    ctex = np.array([10.06, 11.21, 11.50, 11.84, 12.10, 12.29, 12.51, 12.69,
                     12.96, 13.28])
    if temp <= tx[0]:
        hc_mat = hcx[0]
        # Hardness at current temp (SI)
    elif temp >= tx[-1]:
        hc_mat = hcx[-1]
    else:
        hc_mat = interpolate_1d(temp, tx, hcx)
    if temp <= txx[0]:
        cte_mat = ctex[0] * 1e-6
        # coefficient of ther exp at current temp (SI)
    elif temp >= tx[-1]:
        cte_mat = ctex[-1] * 1e-6
    else:
        cte_mat = interpolate_1d(temp, txx, ctex) * 1e-6

    life_mat = 6
    # STLE material code
    life_proc = 7
    # STLE processing code
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    mat_prop = (den_mat,
                # Denisty at current temperature of material.
                em_mat,
                # Elastic modulus at current temperature of material.
                po_mat,
                # Possion ratio at current temperature of material.
                elst_mat,
                # Elastic strain limit at current temperature of material.
                cte_mat,
                # Coefficient of ther exp at current temperature of material.
                tcd_mat,
                # Thermal conductivity at current temperature of material.
                sph_mat,
                # Specific heat at current temperature of material.
                wc_mat,
                # Wear coefficient at current temperature of material.
                hc_mat,
                # Hardness at current temperature of material.
                vms_mat,
                # Von-Mises stress at current temperature of material.
                life_mat,
                # STLE material code of material.
                life_proc
                # STLE processing code of material.
                )

    return mat_prop

@numba.njit(fastmath=False)
def si3n4(curr_temp):
    """Mat properties.

    Parameters
    ----------
    curr_temp: float
        Current temperature.
    curr_pres: float
        Current pressure.

    Returns
    -------
    mat_prop: tuple
        Mat properties.
    """
    mat_name = 'Silicon Nitride (Si3N4)'
    den_mat = 3310.
    # Denisty at current temp (SI)
    em_mat = 3.17e11
    # Elastic modulus at current temp (SI)
    po_mat = 0.23
    # Possion ratio at current temp (SI)
    cte_mat = 3.4e-5
    # coefficient of ther exp at current temp (SI)
    tcd_mat = 27.
    # Thermal conductivity at current temp (SI)
    sph_mat = 711.297
    # Specific heat at current temp (SI)
    elst_mat = 0.002
    # Elastic strain limit at current temp (SI)
    hc_mat = 80.
    # Hardness at current temp (SI)
    wc_mat = 1e-5
    # Wear coefficient at current temp (SI)
    vms_mat = 1.22e9
    # Von-Mises stress at current temp (SI)

    life_mat = 51
    # STLE material code
    life_proc = 1
    # STLE processing code
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    mat_prop = (den_mat,
                # Denisty at current temperature of material.
                em_mat,
                # Elastic modulus at current temperature of material.
                po_mat,
                # Possion ratio at current temperature of material.
                elst_mat,
                # Elastic strain limit at current temperature of material.
                cte_mat,
                # Coefficient of ther exp at current temperature of material.
                tcd_mat,
                # Thermal conductivity at current temperature of material.
                sph_mat,
                # Specific heat at current temperature of material.
                wc_mat,
                # Wear coefficient at current temperature of material.
                hc_mat,
                # Hardness at current temperature of material.
                vms_mat,
                # Von-Mises stress at current temperature of material.
                life_mat,
                # STLE material code of material.
                life_proc
                # STLE processing code of material.
                )

    return mat_prop

@numba.njit(fastmath=False)
def zro2(curr_temp):
    """Mat properties.

    Parameters
    ----------
    curr_temp: float
        Current temperature.
    curr_pres: float
        Current pressure.

    Returns
    -------
    mat_prop: tuple
        Mat properties.
    """
    mat_name = 'Zirconium Oxide (ZrO2) Monoclinic'
    den_mat = 5680.
    # Denisty at current temp (SI)
    em_mat = 2.45e11
    # Elastic modulus at current temp (SI)
    po_mat = 0.23
    # Possion ratio at current temp (SI)
    cte_mat = 7e-6
    # coefficient of ther exp at current temp (SI)
    tcd_mat = 1.675
    # Thermal conductivity at current temp (SI)
    sph_mat = 502.
    # Specific heat at current temp (SI)
    elst_mat = 0.002
    # Elastic strain limit at current temp (SI)
    hc_mat = 80.
    # Hardness at current temp (SI)
    wc_mat = 1e-5
    # Wear coefficient at current temp (SI)
    vms_mat = 1.22e9
    # Von-Mises stress at current temp (SI)

    life_mat = 51
    # STLE material code
    life_proc = 1
    # STLE processing code
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    mat_prop = (den_mat,
                # Denisty at current temperature of material.
                em_mat,
                # Elastic modulus at current temperature of material.
                po_mat,
                # Possion ratio at current temperature of material.
                elst_mat,
                # Elastic strain limit at current temperature of material.
                cte_mat,
                # Coefficient of ther exp at current temperature of material.
                tcd_mat,
                # Thermal conductivity at current temperature of material.
                sph_mat,
                # Specific heat at current temperature of material.
                wc_mat,
                # Wear coefficient at current temperature of material.
                hc_mat,
                # Hardness at current temperature of material.
                vms_mat,
                # Von-Mises stress at current temperature of material.
                life_mat,
                # STLE material code of material.
                life_proc
                # STLE processing code of material.
                )

    return mat_prop

@numba.njit(fastmath=False)
def al2o3(curr_temp):
    """Mat properties.

    Parameters
    ----------
    curr_temp: float
        Current temperature.
    curr_pres: float
        Current pressure.

    Returns
    -------
    mat_prop: tuple
        Mat properties.
    """
    mat_name = 'Aluminium oxide'
    den_mat = 3900.
    # Denisty at current temp (SI)
    em_mat = 3.8e11
    # Elastic modulus at current temp (SI)
    po_mat = 0.24
    # Possion ratio at current temp (SI)
    cte_mat = 8.5e-6
    # coefficient of ther exp at current temp (SI)
    tcd_mat = 34.
    # Thermal conductivity at current temp (SI)
    sph_mat = 858.
    # Specific heat at current temp (SI)
    elst_mat = 0.002
    # Elastic strain limit at current temp (SI)
    hc_mat = 80.
    # Hardness at current temp (SI)
    wc_mat = 1e-5
    # Wear coefficient at current temp (SI)
    vms_mat = 1.22e9
    # Von-Mises stress at current temp (SI)

    life_mat = 51
    # STLE material code
    life_proc = 1
    # STLE processing code
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    mat_prop = (den_mat,
                # Denisty at current temperature of material.
                em_mat,
                # Elastic modulus at current temperature of material.
                po_mat,
                # Possion ratio at current temperature of material.
                elst_mat,
                # Elastic strain limit at current temperature of material.
                cte_mat,
                # Coefficient of ther exp at current temperature of material.
                tcd_mat,
                # Thermal conductivity at current temperature of material.
                sph_mat,
                # Specific heat at current temperature of material.
                wc_mat,
                # Wear coefficient at current temperature of material.
                hc_mat,
                # Hardness at current temperature of material.
                vms_mat,
                # Von-Mises stress at current temperature of material.
                life_mat,
                # STLE material code of material.
                life_proc
                # STLE processing code of material.
                )

    return mat_prop

@numba.njit(fastmath=False)
def titanium(curr_temp):
    """Mat properties.

    Parameters
    ----------
    curr_temp: float
        Current temperature.
    curr_pres: float
        Current pressure.

    Returns
    -------
    mat_prop: tuple
        Mat properties.
    """
    mat_name = 'Titanium-6Al-4V'
    den_mat = 4430.
    # Denisty at current temp (SI)
    po_mat = 0.342
    # Possion ratio at current temp (SI)
    sph_mat = 526.3
    # Specific heat at current temp (SI)
    elst_mat = 0.002
    # Elastic strain limit at current temp (SI)
    wc_mat = 1.0e-5
    # Wear coefficient at current temp (SI)
    hc_mat = 36.
    # Hardness at current temp (SI)
    em_mat = 1.138e11
    # Elastic modulus at current temp (SI)

    temp = curr_temp
    tx = np.array([34.55, 55.46, 86.89, 114.56, 143.86, 293.00, 523.00,
                   773.00])
    ctex = np.array([2.63, 3.80, 5.57, 7.28, 8.21, 8.60, 9.20, 9.70])
    # Temp array for heat capicity & ther cond, the values are in K
    txx = np.array([33.68, 88.72, 143.76, 201.56, 257.16, 294.00])
    tcx = np.array([1.63, 3.24, 4.55, 5.30, 5.99, 6.70])

    if temp <= tx[0]:
        cte_mat = ctex[0] * 1e-6
        # Coefficient of ther exp at current temp (SI)
    elif temp >= tx[-1]:
        cte_mat = ctex[-1] * 1e-6
    else:
        cte_mat = interpolate_1d(temp, tx, ctex) * 1e-6
    if temp <= txx[0]:
        tcd_mat = tcx[0]
        # coefficient of ther exp at current temp (SI)
    elif temp >= tx[-1]:
        tcd_mat = tcx[-1]
    else:
        tcd_mat = interpolate_1d(temp, txx, tcx)
    vms_mat = None
    # Von-Mises stress at current temp (SI)

    life_mat = None
    # STLE material code
    life_proc = None
    # STLE processing code
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    mat_prop = (den_mat,
                # Denisty at current temperature of material.
                em_mat,
                # Elastic modulus at current temperature of material.
                po_mat,
                # Possion ratio at current temperature of material.
                elst_mat,
                # Elastic strain limit at current temperature of material.
                cte_mat,
                # Coefficient of ther exp at current temperature of material.
                tcd_mat,
                # Thermal conductivity at current temperature of material.
                sph_mat,
                # Specific heat at current temperature of material.
                wc_mat,
                # Wear coefficient at current temperature of material.
                hc_mat,
                # Hardness at current temperature of material.
                vms_mat,
                # Von-Mises stress at current temperature of material.
                life_mat,
                # STLE material code of material.
                life_proc
                # STLE processing code of material.
                )

    return mat_prop

@numba.njit(fastmath=False)
def peek(curr_temp):
    """Mat properties.

    Parameters
    ----------
    curr_temp: float
        Current temperature.
    curr_pres: float
        Current pressure.

    Returns
    -------
    mat_prop: tuple
        Mat properties.
    """
    mat_name = 'Peek'
    den_mat = 1440.
    # Denisty at current temp (SI)
    em_mat = 5.86e9
    # Elastic modulus at current temp (SI)
    po_mat = 0.4
    # Possion ratio at current temp (SI)
    cte_mat = 3.06e-5
    # coefficient of ther exp at current temp (SI)
    tcd_mat = 0.245
    # Thermal conductivity at current temp (SI)
    sph_mat = 334.87
    # Specific heat at current temp (SI)
    elst_mat = 0.0129
    # Elastic strain limit at current temp (SI)
    hc_mat = 5.
    # Hardness at current temp (SI)
    wc_mat = 1e-4
    # Wear coefficient at current temp (SI)
    vms_mat = None
    # Von-Mises stress at current temp (SI)

    life_mat = None
    # STLE material code
    life_proc = None
    # STLE processing code
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    mat_prop = (den_mat,
                # Denisty at current temperature of material.
                em_mat,
                # Elastic modulus at current temperature of material.
                po_mat,
                # Possion ratio at current temperature of material.
                elst_mat,
                # Elastic strain limit at current temperature of material.
                cte_mat,
                # Coefficient of ther exp at current temperature of material.
                tcd_mat,
                # Thermal conductivity at current temperature of material.
                sph_mat,
                # Specific heat at current temperature of material.
                wc_mat,
                # Wear coefficient at current temperature of material.
                hc_mat,
                # Hardness at current temperature of material.
                vms_mat,
                # Von-Mises stress at current temperature of material.
                life_mat,
                # STLE material code of material.
                life_proc
                # STLE processing code of material.
                )

    return mat_prop

@numba.njit(fastmath=False)
def polyamide_nylon(curr_temp):
    """Mat properties.

    Parameters
    ----------
    curr_temp: float
        Current temperature.
    curr_pres: float
        Current pressure.

    Returns
    -------
    mat_prop: tuple
        Mat properties.
    """
    mat_name = 'Polyamide - Nylon'
    den_mat = 1140.
    # Denisty at current temp (SI)
    em_mat = 3.3e9
    # Elastic modulus at current temp (SI)
    po_mat = 0.41
    # Possion ratio at current temp (SI)
    cte_mat = 9e-5
    # coefficient of ther exp at current temp (SI)
    tcd_mat = 0.25
    # Thermal conductivity at current temp (SI)
    sph_mat = 1670.
    # Specific heat at current temp (SI)
    elst_mat = 0.01
    # Elastic strain limit at current temp (SI)
    hc_mat = 5.
    # Hardness at current temp (SI)
    wc_mat = 1e-4
    # Wear coefficient at current temp (SI)
    vms_mat = None
    # Von-Mises stress at current temp (SI)

    life_mat = None
    # STLE material code
    life_proc = None
    # STLE processing code
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    mat_prop = (den_mat,
                # Denisty at current temperature of material.
                em_mat,
                # Elastic modulus at current temperature of material.
                po_mat,
                # Possion ratio at current temperature of material.
                elst_mat,
                # Elastic strain limit at current temperature of material.
                cte_mat,
                # Coefficient of ther exp at current temperature of material.
                tcd_mat,
                # Thermal conductivity at current temperature of material.
                sph_mat,
                # Specific heat at current temperature of material.
                wc_mat,
                # Wear coefficient at current temperature of material.
                hc_mat,
                # Hardness at current temperature of material.
                vms_mat,
                # Von-Mises stress at current temperature of material.
                life_mat,
                # STLE material code of material.
                life_proc
                # STLE processing code of material.
                )

    return mat_prop