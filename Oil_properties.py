# -*- coding: utf-8 -*-
"""
Created on Tue Feb 01 8:00:00 2025

@author: Baomaokuan's Chengguo
Program: Bearing Analysis of Mechanical Kinetics-b(V1.0a) Oil_properties
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
def oil_main(oil_type, curr_temp, curr_pres):
    """Choose oil

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
    if oil_type == 0:
        oil_prop = mil_l_7808(curr_temp, curr_pres)
    elif oil_type == 1:
        oil_prop = mil_l_23699(curr_temp, curr_pres)
    elif oil_type == 2:
        oil_prop = mil_l_27502(curr_temp, curr_pres)
    elif oil_type == 3:
        oil_prop = santotrac30(curr_temp, curr_pres)
    elif oil_type == 4:
        oil_prop = santotrac50(curr_temp, curr_pres)
    elif oil_type == 5:
        oil_prop = sae30(curr_temp, curr_pres)

    return oil_prop

###############################################################################
#                              Properties of oil                              #
###############################################################################
@numba.njit(fastmath=False)
def mil_l_7808(curr_temp, curr_pres):
    """Oil properties.

    Parameters
    ----------
    curr_temp: float
        Current temperature.
    curr_pres: float
        Current pressure.

    Returns
    -------
    oil_prop: tuple
        Oil properties.
    """
    lub_name = 'MIL-L-7808'
    vis_mod_name = 'Customized logarithmic relationship'
    trac_alpha = 5.2214e-9
    # Traction pressure-vis coefficient (SI)
    ref_trac_temp = 313.
    ref_trac_vis = 0.11431
    tx = np.array([311., 367., 422., 478.])
    # Temp array for density,heat capicity & ther cond, the values are in K
    dent = np.array([954., 914., 870., 826.])
    #  Values are in kg / m ** 3
    hct = np.array([1.8002178e3, 1.9676798e3, 2.1770074e3, 2.3444696e3])
    tct = np.array([0.1448548, 0.133952, 0.1326534, 0.1313558])

    temp = curr_temp
    ax = math.exp(-3.7048 * math.log(temp) + 22.216)
    bx = math.exp(ax)
    vis_lub = 9.997e-4 * (bx - 0.87)
    # Base viscosity at current temperature (SI)
    den_lub = interpolate_1d(curr_temp, tx, dent)
    # Density at current temperature (SI)
    sp_heat_lub = interpolate_1d(curr_temp, tx, hct)
    # Specific heat at current temp (SI)
    ther_cond_lub = interpolate_1d(curr_temp, tx, tct)
    # Thermal conductivity at current temp (SI)
    vis_coeff_0 = 7.5808e-10 + 1.0742e-6 / (temp - 215.56)
    # Pressure-viscosity coefficient
    vis_coeff_1 = bx * ax * temp * 3.7048 / (bx - 0.87)
    # Temperature-viscosity coefficient(Type 2)
    pres = curr_pres
    vis_lub = vis_lub * math.exp(vis_coeff_0 * pres)
    # Base viscosity at current temperature.
    dvis_lub = vis_lub * vis_coeff_1 / temp ** 2.
    # Viscosity-temperature derivative
    ###########################################################################
    #                    Rheological constant of MilL_7808                    #
    ###########################################################################
    trac_vis = ref_trac_vis * math.exp(0.033723 * (313. - temp))
    # Traction viscosity at current temp(SI)
    trac_beta = 0.041745 * (temp / 313.) ** 0.60873
    # Temperature viscosity coefficient
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    oil_prop = (trac_alpha,
                # Traction pressure-vis coefficient of oil.
                trac_beta,
                # Temperature viscosity coefficient of oil.
                trac_vis,
                # Traction viscosity at current temperature of oil.
                vis_lub,
                # Base viscosity at current temperature of oil.
                den_lub,
                # Density at current temperature of oil.
                sp_heat_lub,
                # Specific heat at current temperature of oil.
                ther_cond_lub,
                # Thermal conductivity at current temperature of oil.
                vis_coeff_0,
                # Pressure-viscosity coefficient of oil.
                vis_coeff_1,
                # Temperature-viscosity coefficient (Type 2) of oil.
                dvis_lub
                # Viscosity-temperature derivative of oil.
                )

    return oil_prop

@numba.njit(fastmath=False)
def mil_l_23699(curr_temp, curr_pres):
    """Oil properties.

    Parameters
    ----------
    curr_temp: float
        Current temperature.
    curr_pres: float
        Current pressure.

    Returns
    -------
    oil_prop: tuple
        Oil properties.
    """
    lub_name = 'MIL-L-23699'
    vis_mod_name = 'Customized logarithmic relationship'
    sp_heat_lub = 2000.
    # Specific heat at current temp (SI)
    trac_alpha = 5.8015e-9
    # Traction pressure-vis coefficient (SI)

    temp = curr_temp
    dtemp = (311.11111111 - temp) / (temp * 311.11111111)
    b1 = 3.6096e3
    b2 = 7.7657e5
    vis_lub = 0.0276 * math.exp(b1 * dtemp + b2 * dtemp ** 2)
    # Base viscosity at current temperature
    vis_coeff_0 = math.exp(1.15607 * math.log(vis_lub) - 17.496) / vis_lub
    # Pressure-viscosity coefficient
    vis_coeff_1 = b1 + 2 * b2 * dtemp
    # Temperature-viscosity coefficient(Type 2)
    ther_cond_lub = 0.14868 - 2.2275e-4 * (temp - 366.67)
    # Thermal conductivity at current temp (SI)
    tx = np.array([233., 300., 367., 422.])
    # Temp array for density,heat capicity & ther cond, the values are in K
    dent = np.array([1016., 966., 922., 884.])
    # Values are in kg / m ** 3
    # f = interpolate.interp1d(tx, dent, kind = 'linear')
    # den_lub_arr = f(curr_temp).reshape([1])
    den_lub = interpolate_1d(curr_temp, tx, dent)
    # Density at current temperature (SI)
    dvis_lub = vis_lub * vis_coeff_1 / temp ** 2
    if curr_pres > 0.:
        vis_lub = vis_lub * math.exp(vis_coeff_0 * curr_pres)
    ###########################################################################
    #                   Rheological constant of MilL_23699                    #
    ###########################################################################
    trac_vis = 0.26529 * math.exp(0.036358 * (313 - temp))
    # Traction viscosity at current temp(SI)
    trac_beta = 0.033398 * (temp / 313) ** 1.3075
    # Temperature viscosity coefficient
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    oil_prop = (trac_alpha,
                # Traction pressure-vis coefficient of oil.
                trac_beta,
                # Temperature viscosity coefficient of oil.
                trac_vis,
                # Traction viscosity at current temperature of oil.
                vis_lub,
                # Base viscosity at current temperature of oil.
                den_lub,
                # Density at current temperature of oil.
                sp_heat_lub,
                # Specific heat at current temperature of oil.
                ther_cond_lub,
                # Thermal conductivity at current temperature of oil.
                vis_coeff_0,
                # Pressure-viscosity coefficient of oil.
                vis_coeff_1,
                # Temperature-viscosity coefficient (Type 2) of oil.
                dvis_lub
                # Viscosity-temperature derivative of oil.
                )

    return oil_prop

@numba.njit(fastmath=False)
def mil_l_27502(curr_temp, curr_pres):
    """Oil properties.

    Parameters
    ----------
    curr_temp: float
        Current temperature.
    curr_pres: float
        Current pressure.

    Returns
    -------
    oil_prop: tuple
        Oil properties.
    """
    lub_name = 'MIL-L-27502 (MCS 1780)'
    sp_heat_lub = 2000.
    # Specific heat at current temp (SI)
    trac_alpha = 5.8015e-9
    # Traction pressure-vis coefficient (SI)
    den_lub = 950.
    # Density at current temperature (SI)

    temp = curr_temp
    dtemp = (311.11111111 - temp) / (temp * 311.11111111)
    b1 = 3.7108e3
    b2 = 7.3904e5
    vis_lub = 0.0276 * math.exp(b1 * dtemp + b2 * dtemp ** 2)
    # Base viscosity at current temperature
    vis_coeff_0 = math.exp(1.15607 * math.log(vis_lub) - 17.496) / vis_lub
    # Pressure-viscosity coefficient
    vis_coeff_1 = b1 + 2 * b2 * dtemp
    # Temperature-viscosity coefficient (Type 2)
    dvis_lub = vis_lub * vis_coeff_1 / temp ** 2
    # Viscosity-temperature derivative
    ther_cond_lub = 0.14868 - 2.2275e-04 * (temp - 366.67)
    # Thermal conductivity at current temp (SI)
    if curr_pres > 0:
        vis_lub = vis_lub * math.exp(vis_coeff_0 * curr_pres)
    ###########################################################################
    #                   Rheological constant of MilL_27052                    #
    ###########################################################################
    trac_vis = 0.21655 * math.exp(0.039221 * (313 - temp))
    # Traction viscosity at current temp (SI)
    trac_beta = 0.032577 * (temp / 313) ** 1.0723
    # Temperature viscosity coefficient (SI)
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    oil_prop = (trac_alpha,
                # Traction pressure-vis coefficient of oil.
                trac_beta,
                # Temperature viscosity coefficient of oil.
                trac_vis,
                # Traction viscosity at current temperature of oil.
                vis_lub,
                # Base viscosity at current temperature of oil.
                den_lub,
                # Density at current temperature of oil.
                sp_heat_lub,
                # Specific heat at current temperature of oil.
                ther_cond_lub,
                # Thermal conductivity at current temperature of oil.
                vis_coeff_0,
                # Pressure-viscosity coefficient of oil.
                vis_coeff_1,
                # Temperature-viscosity coefficient (Type 2) of oil.
                dvis_lub
                # Viscosity-temperature derivative of oil.
                )

    return oil_prop

@numba.njit(fastmath=False)
def santotrac30(curr_temp, curr_pres):
    """Oil properties.

    Parameters
    ----------
    curr_temp: float
        Current temperature.
    curr_pres: float
        Current pressure.

    Returns
    -------
    oil_prop: tuple
        Oil properties.
    """
    lub_name = 'Santotrac 30'
    sp_heat_lub = 2000.
    # Specific heat at current temp (SI)
    trac_alpha = 9.4275e-9
    # Traction pressure-vis coefficient (SI)
    den_lub = 850.
    # Density at current temperature (SI)

    temp = curr_temp
    vis_coeff_0 = 2.9008e-8 * math.exp(-0.0083178 * (temp - 277.78))
    # Pressure-viscosity coefficient
    dn = 0.817 + 6.66e-4 * (422.2222 - temp)
    aa = 10 ** (-4.1044 * math.log10(temp) + 10.305)
    vis_lub = 0.001 * dn * (10 ** aa - 0.8)

    ti1 = temp - 1.
    dn1 = 0.817 + 6.66e-04 * (422.2222 - ti1)
    aa1 = 10 ** (-4.1044 * math.log10(ti1) + 10.305)
    vis1 = 0.001 * dn1 * (10 ** aa1 - 0.8)
    ti2 = ti1 + 1
    dn2 = 0.817 + 6.66e-4 * (422.2222 - ti2)
    aa2 = 10. ** (-4.1044 * math.log10(ti2) + 10.305)
    vis2 = 0.001 * dn2 * (10 ** aa2 - 0.8)
    vis_coeff_1 = math.log(vis1 / vis2) * (ti2 ** 2 - 2 * ti2) / 2
    # Temperature-viscosity coefficient (Type 2)
    dvis_lub = vis_lub * vis_coeff_1 / temp ** 2 # Viscosity-temperature derivative
    ther_cond_lub = 0.10904 - 1.7091e-5 * abs(temp - 311.1111111) ** 1.322
    # Thermal conductivity at current temp (SI)
    if curr_pres > 0:
        vis_lub = vis_lub * math.exp(vis_coeff_0 * curr_pres)
    ###########################################################################
    #                  Rheological constant of Santotrac 30                   #
    ###########################################################################
    trac_vis = 0.40063 * math.exp(0.050459 * (313 - temp))
    # Traction viscosity at current temp (SI)
    trac_beta = 0.034104 * (temp / 313) ** 1.093
    # Temperature viscosity coefficient (SI)
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    oil_prop = (trac_alpha,
                # Traction pressure-vis coefficient of oil.
                trac_beta,
                # Temperature viscosity coefficient of oil.
                trac_vis,
                # Traction viscosity at current temperature of oil.
                vis_lub,
                # Base viscosity at current temperature of oil.
                den_lub,
                # Density at current temperature of oil.
                sp_heat_lub,
                # Specific heat at current temperature of oil.
                ther_cond_lub,
                # Thermal conductivity at current temperature of oil.
                vis_coeff_0,
                # Pressure-viscosity coefficient of oil.
                vis_coeff_1,
                # Temperature-viscosity coefficient (Type 2) of oil.
                dvis_lub
                # Viscosity-temperature derivative of oil.
                )

    return oil_prop

@numba.njit(fastmath=False)
def santotrac50(curr_temp, curr_pres):
    """Oil properties.

    Parameters
    ----------
    curr_temp: float
        Current temperature.
    curr_pres: float
        Current pressure.

    Returns
    -------
    oil_prop: tuple
        Oil properties.
    """
    lub_name = 'Santotrac 50'
    sp_heat_lub = 2000.
    # Specific heat at current temp (SI)
    trac_alpha = 7.2519e-9
    # Traction pressure-vis coefficient (SI)
    den_lub = 850.
    # Density at current temperature (SI)

    temp = curr_temp
    vis_coeff_0 = 3.916e-8 * math.exp(-0.0083178 * (temp - 277.78))
    # Pressure-viscosity coefficient
    dn = 0.82 + 6.21e-4 * (422.2222 - temp)
    aa = 10 ** (-3.7452 * math.log10(temp) + 9.5363)
    vis_lub = 0.001 * dn * (10 ** aa - 0.8)
    ti = temp - 1.
    dn = 0.82 + 6.21e-4 * (422.2222 - ti)
    aa = 10. ** (-3.7452 * math.log10(ti) + 9.5363)
    vis1 = 0.001 * dn * (10 ** aa - 0.8)
    ti = ti + 1
    dn = 0.82 + 6.21e-4 * (422.2222 - ti)
    aa = 10. ** (-3.7452 * math.log10(ti) + 9.5363)
    vis2 = 0.001 * dn * (10 ** aa - 0.8)
    vis_coeff_1 = math.log(vis1 / vis2) * (ti ** 2 - 2 * ti) / 2
    dvis_lub = (vis_lub * vis_lub / temp ** 2)
    # Viscosity-temperature derivative
    ther_cond_lub = (0.10385 - 5.6078e-7 * (abs(temp - 311.1111111)) ** 2)
    # Thermal conductivity at current temp (SI)
    if curr_pres > 0:
        vis_lub = vis_lub * math.exp(vis_coeff_0 * curr_pres)
    ###########################################################################
    #                  Rheological constant of Santotrac 50                   #
    ###########################################################################
    trac_vis = 10.076 * math.exp(0.044097 * (313 - temp))
    # Traction viscosity at current temp (SI)
    trac_beta = 0.020794 * (temp / 313) ** 6.2743
    # Temperature viscosity coefficient (SI)
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    oil_prop = (trac_alpha,
                # Traction pressure-vis coefficient of oil.
                trac_beta,
                # Temperature viscosity coefficient of oil.
                trac_vis,
                # Traction viscosity at current temperature of oil.
                vis_lub,
                # Base viscosity at current temperature of oil.
                den_lub,
                # Density at current temperature of oil.
                sp_heat_lub,
                # Specific heat at current temperature of oil.
                ther_cond_lub,
                # Thermal conductivity at current temperature of oil.
                vis_coeff_0,
                # Pressure-viscosity coefficient of oil.
                vis_coeff_1,
                # Temperature-viscosity coefficient (Type 2) of oil.
                dvis_lub
                # Viscosity-temperature derivative of oil.
                )

    return oil_prop

@numba.njit(fastmath=False)
def sae30(curr_temp, curr_pres):
    """Oil properties.

    Parameters
    ----------
    curr_temp: float
        Current temperature.
    curr_pres: float
        Current pressure.

    Returns
    -------
    oil_prop: tuple
        Oil properties.
    """
    lub_name='Mobil DTE (SAE 30)'
    ther_cond_lub = 0.096082
    # Thermal conductivity at current temp (SI)
    den_lub = 850.
    # Density at current temperature (SI)
    sp_heat_lub = 2000.
    ref_tv = 311.11111111
    ref_vis = 0.07166
    vis_coeff_0 = 1.9e-8
    # Pressure-viscosity coefficient (SI)
    vis_coeff_1 = 4104.3
    # Temperature-viscosity coefficient(Type 2)
    ref_trac_temp = 313.
    ref_trac_vis = 0.28407
    trac_alpha = 7.2516e-9
    # Traction pressure-vis coefficient (SI)
    temp = curr_temp
    dt = (ref_tv - curr_temp) / (ref_tv * curr_temp)
    vis_lub = ref_vis * math.exp(vis_coeff_0 * curr_pres + vis_coeff_1 * dt)
    # Base viscosity at current temperature (SI)
    dvis_lub = vis_lub * vis_coeff_1 / curr_temp ** 2
    # Viscosity-temperature derivative
    ###########################################################################
    #               Rheological constant of Mobil DTE (SAE 30)                #
    ###########################################################################
    trac_vis = ref_trac_vis * math.exp(0.023743 * (ref_trac_temp - temp))
    # Traction viscosity at current temp (SI)
    trac_beta = 0.028939 * (temp / ref_trac_temp) ** 2.1402
    # Temperature viscosity coefficient (SI)
    ###########################################################################
    #                              Store result                               #
    ###########################################################################
    oil_prop = (trac_alpha,
                # Traction pressure-vis coefficient of oil.
                trac_beta,
                # Temperature viscosity coefficient of oil.
                trac_vis,
                # Traction viscosity at current temperature of oil.
                vis_lub,
                # Base viscosity at current temperature of oil.
                den_lub,
                # Density at current temperature of oil.
                sp_heat_lub,
                # Specific heat at current temperature of oil.
                ther_cond_lub,
                # Thermal conductivity at current temperature of oil.
                vis_coeff_0,
                # Pressure-viscosity coefficient of oil.
                vis_coeff_1,
                # Temperature-viscosity coefficient (Type 2) of oil.
                dvis_lub
                # Viscosity-temperature derivative of oil.
                )

    return oil_prop