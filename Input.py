# -*- coding: utf-8 -*-
"""
Created on Tue Feb 01 8:00:00 2025

@author: Baomaokuan's Chengguo
Program: Bearing Analysis of Mechanical Kinetics-b(V1.0a) Input
"""

###############################################################################
#                                Input library                                #
###############################################################################
import math
import pickle
import numpy as np

from Mat_properties import *
from Oil_properties import *
###############################################################################
#                                  Read data                                  #
###############################################################################
with open('Data.txt', 'r', encoding='utf-8', errors='ignore') as file:
    data = []
    for line in file:
        line = line.rstrip('\n')
        rows = line.split(';')
        row_data = []
        for row in rows:
            elements = row.split(',')
            for i, element in enumerate(elements):
                element = element.strip()
                if i == 0 and element:
                    row_data.append(element)
                if i != 0 and element:
                    row_data.append(float(element))
        data.append(row_data)
###############################################################################
#                                 Input data                                  #
###############################################################################
brg_type = float(data[0][1])
# Bearing type
###############################################################################
#                                  Heat data                                  #
###############################################################################
"""
Base heat Properties
"""
ini_temp_r = float(data[1][1])
# Initial room temperature(K)
ini_temp_h = float(data[1][2])
# Initial housing bulk temperature(K)
ini_temp_s = float(data[1][3])
# Initial shafting bulk temperature(K)
ini_temp_o = float(data[1][4])
# Initial outer race temperature(K)
ini_temp_i = float(data[1][5])
# Initial inner race temperature(K)
ini_temp_b = float(data[1][6])
# Initial ball temperature(K)
ini_temp_c = float(data[1][7])
# Initial cage temperature(K)
###############################################################################
###############################################################################
#                                  Ball data                                  #
###############################################################################
"""
Geometry Properties
"""
D_b = float(data[2][1])
# Ball diameter
R_b = D_b / 2
# Ball radius
n = int(data[2][2])
# Ball number
"""
Material Properties
"""
mat_type_b = int(data[2][3])
mat_prop_b = mat_main(mat_type_b, ini_temp_b)
den_b = mat_prop_b[0]
# Density of ball
E_b = mat_prop_b[1]
# Elastic Modulus of ball
po_b = mat_prop_b[2]
# Poison number of ball
elas_stra_limt_b = mat_prop_b[3]
# Ball stress limit defined by modulus
coeff_ther_exp_b = mat_prop_b[4]
# Coefficient of ball thermal expansion
ther_cond_b = mat_prop_b[5]
# Thermal conductivity of ball
spec_heat_b = mat_prop_b[6]
# Specific heat of ball
wear_coff_b = mat_prop_b[7]
# Wear cofficient of ball
hard_coff_b = mat_prop_b[8]
# Hardness of ball
von_mises_stress_b = mat_prop_b[9]
# Von mises stress limited of ball
mat_fac_type_b = mat_prop_b[10]
# STLE material codes of ball
proc_fac_type_b = mat_prop_b[11]
# STLE processing codes of ball
rms_b = float(data[2][4])
# Roughness of ball
###############################################################################
###############################################################################
#                             Other bearing data                              #
###############################################################################
D_m = float(data[3][1])
# Bearing pitch diameter
R_m = D_m / 2
# Bearing pitch radius
free_con_ang = float(data[3][2]) * math.pi / 180
# Free contact angle with ball / race
###############################################################################
###############################################################################
#                               Outer race data                               #
###############################################################################
"""
Geometry Properties
"""
D_o_u = float(data[4][1])
# Outer diameter of outer race
R_o_u = D_o_u/2
# Outer radius of outer race
D_o_m = float(data[4][2])
# Bearing outer flange diameter
R_o_m = D_o_m / 2
# Bearing outer flange radius
f_o = float(data[4][3])
# Groove curvature of outer race
D_o_d = (D_m - 2 * (f_o - 0.5) * D_b * math.cos(free_con_ang)) + 2 * f_o * D_b
# Inner groove bottom diameter of Bearing
R_o_d = D_o_d/2
# Inner groove bottom radius of Bearing
W_o = float(data[4][4])
# Width of outer race
Shim_thknss_o = float(data[4][5])
# Shim thickness for split outer race
sam_o = Shim_thknss_o / (D_b * (2 * f_o - 1))
cam_o = math.sqrt(1 - sam_o ** 2)
shim_ang_o = math.atan(sam_o / cam_o)
# shim angle in radians
"""
Material Properties
"""
mat_type_o = int(data[4][6])
mat_prop_o = mat_main(mat_type_o, ini_temp_o)
den_o = mat_prop_o[0]
# Density of ball in dir x
E_o = mat_prop_o[1]
# Elastic Modulus of outer race
po_o = mat_prop_o[2]
# Poison number of outer race
elas_stra_limt_o = mat_prop_o[3]
# Outer race stress limit defined by modulus
coeff_ther_exp_o = mat_prop_o[4]
# Coefficient of outer race thermal expansion
ther_cond_o = mat_prop_o[5]
# Thermal conductivity of outer race
spec_heat_o = mat_prop_o[6]
# Specific heat of outer race
wear_coff_o = mat_prop_o[7]
# Wear cofficient of outer race
hard_coff_o = mat_prop_o[8]
# Hardness of outer race
von_mises_stress_o = mat_prop_o[9]
# Von mises stress limited of outer race
mat_fac_type_o = mat_prop_o[10]
# STLE material codes of outer race
proc_fac_type_o = mat_prop_o[11]
# STLE processing codes of outer race
rms_o = float(data[4][7])
# Roughness of outer race
"""
Physics Properties
"""
dmpg_b_o = float(data[4][8])
# Rolling element to or damping ratio
"""
Else Properties
"""
rpm_o = float(data[4][9])
# Angular velocity of outer race(rpm)
mis_o_y = float(data[4][10]) * math.pi / 180
# Outer race misalignment of y
mis_o_z = float(data[4][11]) * math.pi / 180
# Outer race misalignment of z
k_tra_o_x = int(data[4][12])
# Outer race transcation coffient in direction x(must be 0)
k_tra_o_y = int(data[4][13])
# Outer race transcation coffient in direction y(must be 0)
k_tra_o_z = int(data[4][14])
# Outer race transcation coffient in direction z(must be 0)
k_rot_o_x = int(data[4][15])
# Outer race rotation coffient in direction x(0 or 1)
k_rot_o_y = int(data[4][16])
# Outer race rotation coffient in direction y(0 or 1)
k_rot_o_z = int(data[4][17])
# Outer race rotation coffient in direction z(0 or 1)
sub_f_o_x = int(data[4][18])
# Dynamic displacement constraint on outer race in direction x
sub_f_o_y = int(data[4][19])
# Dynamic displacement constraint on outer race in direction y
sub_f_o_z = int(data[4][20])
# Dynamic displacement constraint on outer race in direction z
sub_m_o_x = int(data[4][21])
# Dynamic rotational constraint on outer race in direction x
sub_m_o_y = int(data[4][22])
# Dynamic rotational constraint on outer race in direction y
sub_m_o_z = int(data[4][23])
# Dynamic rotational constraint on outer race in direction z
###############################################################################
###############################################################################
#                               Inner race data                               #
###############################################################################
"""
Geometry Properties
"""
D_i_d = float(data[5][1])
# Inner diameter of inner race
R_i_d = D_i_d/2
# Inner radius of inner race
D_i_m = float(data[5][2])
# Bearing inner flange diameter
R_i_m = D_i_m/2
# Bearing inner flange diameter
f_i = float(data[5][3])
# Groove curvature of inner race
D_i_u = (D_m + 2 * (f_i - 0.5) * D_b * math.cos(free_con_ang)) - 2 * f_i * D_b
# Inner groove bottom diameter of Bearing
R_i_u = D_i_u / 2
# Inner groove bottom radius of Bearing
W_i = float(data[5][4])
# Width of inner race
Shim_thknss_i = float(data[5][5])
# Shim thickness for split inner race
sam_i = Shim_thknss_i / (D_b * (2 * f_i - 1))
cam_i = math.sqrt(1 - sam_i ** 2)
shim_ang_i = math.atan(sam_i / cam_i)
# shim angle in radians
"""
Material Properties
"""
mat_type_i = int(data[5][6])
mat_prop_i = mat_main(mat_type_i, ini_temp_i)
den_i = mat_prop_i[0]
# Density of ball in dir x
E_i = mat_prop_i[1]
# Elastic Modulus of inner race
po_i = mat_prop_i[2]
# Poison number of inner race
elas_stra_limt_i = mat_prop_i[3]
# Inner race stress limit defined by modulus
coeff_ther_exp_i = mat_prop_i[4]
# Coefficient of inner race thermal expansion
ther_cond_i = mat_prop_i[5]
# Thermal conductivity of inner race
spec_heat_i = mat_prop_i[6]
# Specific heat of inner race
wear_coff_i = mat_prop_i[7]
# Wear cofficient of inner race
hard_coff_i = mat_prop_i[8]
# Hardness of inner race
von_mises_stress_i = mat_prop_i[9]
# Von mises stress limited of inner race
mat_fac_type_i = mat_prop_i[10]
# STLE material codes of inner race
proc_fac_type_i = mat_prop_i[11]
# STLE processing codes of inner race
rms_i = float(data[5][7])
# Roughness of inner race
"""
Physics Properties
"""
dmpg_b_i = float(data[5][8])
# Rolling element to ir damping ratio
"""
Else Properties
"""
rpm_i = float(data[5][9])
# Angular velocity of inner race(rpm)
mis_i_y = float(data[5][10]) * math.pi / 180
# Inner race misalignment of y
mis_i_z = float(data[5][11]) * math.pi / 180
# Inner race misalignment of z
k_tra_i_x = int(data[5][12])
# Inner race transcation coffient in direction x(must be 1)
k_tra_i_y = int(data[5][13])
# Inner race transcation coffient in direction y(must be 1)
k_tra_i_z = int(data[5][14])
# Inner race transcation coffient in direction z(must be 1)
k_rot_i_x = int(data[5][15])
# Inner race rotation coffient in direction x(0 or 1)
k_rot_i_y = int(data[5][16])
# Inner race rotation coffient in direction y(0 or 1)
k_rot_i_z = int(data[5][17])
# Inner race rotation coffient in direction z(0 or 1)
sub_f_i_x = int(data[5][18])
# Dynamic displacement constraint on inner race in direction x
sub_f_i_y = int(data[5][19])
# Dynamic displacement constraint on inner race in direction y
sub_f_i_z = int(data[5][20])
# Dynamic displacement constraint on inner race in direction z
sub_m_i_x = int(data[5][21])
# Dynamic rotational constraint on inner race in direction x
sub_m_i_y = int(data[5][22])
# Dynamic rotational constraint on inner race in direction y
sub_m_i_z = int(data[5][23])
# Dynamic rotational constraint on inner race in direction z
###############################################################################
###############################################################################
#                                  Cage data                                  #
###############################################################################
n_cseg = int(data[6][1])
# Number of cage segments in the bearing
"""
Geometry Properties
"""
D_c_u = float(data[6][2])
# Outside Center diameter of cage
R_c_u = D_c_u / 2
# Outside Center radius of cage
D_c_d = float(data[6][3])
# Inside Center diameter of cage
R_c_d = D_c_d / 2
# Inside Center radius of cage
D_c_g = float(data[6][4])
# Diameter of guide
R_c_g = D_c_g / 2
# Radius of guide
D_p_u = float(data[6][5])
# Up pocket diameter clearance
R_p_u = D_p_u / 2
# Up pocket radius clearance
D_p_d = float(data[6][6])
# Down pocket diameter clearance
R_p_d = D_p_d / 2
# Down pocket radius clearance
W_c = float(data[6][7])
# Width of cage
Sg = W_c / 2
# Half width of cage
Cg_u = float(data[6][8])
# Up guide clearance
Cg_d = float(data[6][9])
# Down guide clearance
Bg_l = float(data[6][10])
# Left Guide wideth
Bg_r = float(data[6][11])
# Right Guide wideth
"""
Material Properties
"""
mat_type_c = int(data[6][12])
mat_prop_c = mat_main(mat_type_c, ini_temp_c)
den_c = mat_prop_c[0]
# Density of cage
E_c = mat_prop_c[1]
# Elastic Modulus of cage
po_c = mat_prop_c[2]
# Poison number of cage
elas_stra_limt_c = mat_prop_c[3]
# Cage race stress limit defined by modulus
coeff_ther_exp_c = mat_prop_c[4]
# Coefficient of cage thermal expansion
ther_cond_c = mat_prop_c[5]
# Thermal conductivity of cage
spec_heat_c = mat_prop_c[6]
# Specific heat heat of cage
wear_coff_c = mat_prop_c[7]
# Wear cofficient of cage
hard_coff_c = mat_prop_c[8]
# Hardness of cage
von_mises_stress_c = mat_prop_c[9]
# Von mises stress limited of cage
rms_c = float(data[6][13])
# Roughness of cage
"""
Physics Properties
"""
dmpg_b_c = float(data[6][14])
# Rolling element to cage damping ratio
dmpg_c_r = float(data[6][15])
# Cage to race damping ratio
"""
Else Properties
"""
poc_lub_film = float(data[6][16])
# Maximum lubricant film in the cage pocket
poc_type = int(data[6][17])
# Pocket shape of cage
if poc_type == 0 or poc_type == 2 or poc_type == 3:
    Poc_cls = 0.
else:
    Poc_cls = float(data[6][18])
cage_freedom = int(data[6][19])
# Freedom number of the cage
c_r_guide_type = int(data[6][20])
# Motion of guide
int_sol = int(data[6][21])
# interpolate hydrodynamic solutions
"""
Some Initial Parameters
"""
wv_ratio = float(data[6][22])
# Initial cage whirl velocity / angular velocity
av_ratio = float(data[6][23])
# Initial cage angular velocity / the epicyclic value
cage_mass_cen_x = float(data[6][24])
# Cage mass center in dir x
cage_mass_cen_y = float(data[6][25])
# Cage mass center in dir y
cage_mass_cen_z = float(data[6][26])
# Cage mass center in dir z
###############################################################################
###############################################################################
#                                   Oil data                                  #
###############################################################################
"""
Base oil Properties
"""
oil_type = int(data[7][1])
# 0 for MIL-L-7808, 1 for MIL-L-23699, 2 for MIL-L-27502(MCS 1780),
# 3 for Santotrac 30, 4 for Santotrac50, 5 for Mobil DTE(SAE 30)
oil_prop = oil_main(oil_type, ini_temp_r, 0)
trac_alpha = oil_prop[0]
# Traction pressure-vis coefficient
trac_beta = oil_prop[1]
# Temperature viscosity coefficient
trac_vis = oil_prop[2]
# Traction viscosity at current temp
vis_lub = oil_prop[3]
# Base viscosity at current temperature
den_lub = oil_prop[4]
# Density at current temperature
spec_heat_lub = oil_prop[5]
# Specific heat at current temp
ther_cond_lub = oil_prop[6]
# Thermal conductivity at current temp
dvis_lub = oil_prop[7]
vis_coeff_0 = oil_prop[8]
# Pressure-viscosity coefficient
vis_coeff_1 = oil_prop[9]
# Temperature-viscosity coefficient(Type 2)
"""
Ball / race oil properties
"""
b_r_lub_type = int(data[7][2])
# Lubrication type between ball and race
k_b_r_trac_type = int(data[7][3])
# Traction type between ball and race
if b_r_lub_type == 0:
    kai0_0 = float(data[7][4])
    # Traction coefficient at zero slip
    kaim_0 = float(data[7][5])
    # Maximum traction coefficient
    kaiin_0 = float(data[7][6])
    # Traction coefficient at infinite slip
    um_0 = float(data[7][7])
    # Slip velocity corresponding to maximum traction
else:
    kai0_0 = float(data[7][4])
    kaim_0 = float(data[7][5])
    kaiin_0 = float(data[7][6])
    um_0 = float(data[7][7])
"""
Ball / cage and cage / race oil properties
"""
b_c_lub_type = int(data[7][8])
# Lubrication type between cage and ball
k_b_c_trac_type = int(data[7][9])
# Traction type between ball and cage
if b_c_lub_type == 0:
    f_b_c = float(data[7][10])
    # Traction coeff between ball and cage
else:
    f_b_c = float(data[7][10])
c_r_lub_type = int(data[7][11])
# Lubrication type between cage and race
k_c_r_trac_type = int(data[7][12])
# Traction type between cage and race
if c_r_lub_type == 0:
    f_c_r = float(data[7][13])
    # Traction coeff between cage and race
else:
    f_c_r = float(data[7][13])
"""
Ball / ball oil properties
"""
if n_cseg <= 0:
    k_b_b_trac_type = int(data[7][14])
    # Traction type between ball and ball
    b_b_limt_film = float(data[7][15])
    # Critical film thickness between ball / ball
    f_b_b = float(data[7][16])
    # Traction coeff between ball and ball
else:
    k_b_b_trac_type = 0
    b_b_limt_film = 0.
    f_b_b = 0.
"""
Other properties
"""
str_parm = float(data[7][17])
# Starvation parameter between ball and race
b_o_limt_film = (rms_b ** 2 + rms_o ** 2) ** 0.5
# Critical film thickness between ball and outer race
b_i_limt_film = (rms_b ** 2 + rms_i ** 2) ** 0.5
# Critical film thickness between ball and inner race
b_c_limt_film = float(data[7][18])
# Critical film thickness between cage and ball
c_r_limt_film = float(data[7][19])
# Critical film thickness between cage and race (Suggest 1e-6 to 5e-6)
# only for ball bearing
"""
Churning properties
"""
k_chrn_type = int(data[7][20])
# Churning for ball and cage
# 1 for custom, 2 for LOX, 3 for LH2, 4 for LN2, 5 for CH4, 6 for RP1,
# 7 for RP2, 8 for JP10, 9 for JP8-3638, 10 for JP8-4658
if k_chrn_type <= 0:
    den_ratio = 0.
    # Density ratio for churning and drag effects
    vis_ratio = 0.
    # Viscosity ratio of churning and drag effects
    chrn_pres = 0.
    # Churning media pressure
    ini_temp_chrn = 0.
    # Churning media temperature
elif k_chrn_type == 1:
    den_ratio = float(data[7][21])
    vis_ratio = float(data[7][22])
    chrn_pres = float(data[7][23])
    ini_temp_chrn = float(data[7][24])
###############################################################################
###############################################################################
#                            House and shaft data                             #
###############################################################################
"""
Some else parameters of ball bearing
"""
D_h = float(data[8][1])
# House outside diameter
D_s = float(data[8][2])
# Shaft inside diameter
Fit_o_h = float(data[8][3])
# Diametral shrink fit allowance on outer race
Fit_i_s = float(data[8][4])
# Diametral shrink fit allowance on inner race
"""
Material Properties
"""
mat_type_h = int(data[8][5])
mat_prop_h = mat_main(mat_type_h, ini_temp_h)
den_h = mat_prop_h[0]
# Density of ball in dir x
E_h = mat_prop_h[1]
# Elastic Modulus of house
po_h = mat_prop_h[2]
# Poison number of house
elas_stra_limt_h = mat_prop_h[3]
# House stress limit defined by modulus
coeff_ther_exp_h = mat_prop_h[4]
# Coefficient of house thermal expansion
ther_cond_h = mat_prop_h[5]
# Thermal conductivity of house
spec_heat_h = mat_prop_h[6]
# Specific heat of house
wear_coff_h = mat_prop_h[7]
# Wear cofficient of house
hard_coff_h = mat_prop_h[8]
# Hardness of house
von_mises_stress_h = mat_prop_h[9]
# Von mises stress limited of house

mat_type_s = int(data[8][6])
mat_prop_s = mat_main(mat_type_s, ini_temp_s)
den_s = mat_prop_s[0]
# Density of ball in dir x
E_s = mat_prop_s[1]
# Elastic Modulus of shaft
po_s = mat_prop_s[2]
# Poison number of shaft
elas_stra_limt_s = mat_prop_s[3]
# Shaft stress limit defined by modulus
coeff_ther_exp_s = mat_prop_s[4]
# Coefficient of shaft thermal expansion
ther_cond_s = mat_prop_s[5]
# Thermal conductivity of shaft
spec_heat_s = mat_prop_s[6]
# Specific heat of shaft
wear_coff_s = mat_prop_s[7]
# Wear cofficient of shaft
hard_coff_s = mat_prop_s[8]
# Hardness of shaft
von_mises_stress_s = mat_prop_s[9]
# Von mises stress limited of shaft
###############################################################################
###############################################################################
#                             Moving frame data                               #
###############################################################################
"""
Moving frame parameters
"""
k_brg_mov = int(data[9][1])
if k_brg_mov <= 0:
    R_brg_orb = 0.
    # Radius of orbit in which bearing center travels.
    ini_brg_ang_pos = 0 * math.pi / 180
    # Initial angular position of bearing center.
    brg_mov_dir = 0.
    # Direction of bearing center travels.
    brg_ang_vel = 0.
    # Angular velocity at which bearing center rotates.
    brg_load_frac_o = 0.
    # Fraction of inertial load exerted on the outer race, due to base bearing 
    # rame rotation, to be transmitted to the bearing.
    brg_load_frac_i = 0.
    # Fraction of inertial load exerted on the inner race, due to base bearing
    # frame rotation, to be transmitted to the bearing.
else:
    R_brg_orb = float(data[9][2])
    ini_brg_ang_pos = float(data[9][3]) * math.pi / 180
    brg_mov_dir = float(data[9][4])
    brg_ang_vel = float(data[9][5])
    brg_load_frac_o = float(data[9][6])
    brg_load_frac_i = float(data[9][7])
###############################################################################
###############################################################################
#                                Gravity data                                 #
###############################################################################
"""
Gravity and vectors
"""
g = float(data[10][1])
# Acceleration due to gravity
G_vec_0 = float(data[10][2])
# X-comp of the acceleration
G_vec_1 = float(data[10][3])
# Y-comp of the acceleration
G_vec_2 = float(data[10][4])
# Z-comp of the acceleration
###############################################################################
###############################################################################
#                             Bearing life data                               #
###############################################################################
k_life_freq = int(data[11][1])
# Frequency of time steps for fatigue life calculation
if k_life_freq <= 0:
    k_life_cons = 0.
    # model constants for fatigue life
    res_stress_o = 0.
    # Residual stress in outer race(>0, Compressive residual stress
    #                               <0, Tensile residual stress)
    res_stress_i = 0.
    # Residual stress in outer race(>0, Compressive residual stress
    #                               <0, Tensile residual stress)
    str_limt_fac = 0.
    # Limit stress modifier for IH life model
    # Valid values are zero or greater
    # = 0, No stress limit. Model converges to LP life  with 
    # maximum octahedral shear stress as the failure stress
    # = 1.00(Default), Octahedral shear stress limit corresponds 
    # to von-Mises stress of the material
    # = 1.28(ISO 281), Octahedral shear stress limit determined 
    # from limiting race contact stress of 1.50 GPa
    # = x, Octahedral shear stress lmit is x times the default value 
    # corresponding to facStrLimit =1
else:
    k_life_cons = float(data[11][2])
    res_stress_o = float(data[11][3])
    res_stress_i = float(data[11][4])
    str_limt_fac = float(data[11][5])
###############################################################################
###############################################################################
#                          Irregular geometry data                            #
###############################################################################
"""
Ball irregular geometry
"""
k_ip_type_b = int(data[12][1])
# Arbitrary user supplied ball inertial parameters
if k_ip_type_b <= 0:
    var_num_b = 0
    # Number of the ball imperfections
    var_inr_b = 1
    # Interval of the ball imperfections (must >= 1)
    m_b = den_b * math.pi * D_b ** 3 / 6
    # Mass of ball
    I_b_x = m_b * D_b ** 2 / 10
    # Interia of ball in dir x
    I_b_y = I_b_x
    # Interia of ball in dir y
    I_b_z = I_b_x
    # Interia of ball in dir z
    geo_cen_b_x = 0.
    # Ball geometric center in dir x
    geo_cen_b_y = 0.
    # Ball geometric center in dir y
    geo_cen_b_z = 0.
    # Ball geometric center in dir z
    deg_b_x = 0 * math.pi / 180
    # Ball geo center to principal triad in dir x
    deg_b_y = 0 * math.pi / 180
    # Ball geo center to principal triad in dir y
    deg_b_z = 0 * math.pi / 180
    # Ball geo center to principal triad in dir z
else:
    var_num_b = int(data[12][2])
    var_inr_b = int(data[12][3])
    m_b = float(data[12][4])
    I_b_x = float(data[12][5])
    I_b_y = float(data[12][6])
    I_b_z = float(data[12][7])
    geo_cen_b_x = float(data[12][8])
    geo_cen_b_y = float(data[12][9])
    geo_cen_b_z = float(data[12][10])
    deg_b_x = float(data[12][11]) * math.pi / 180
    deg_b_y = float(data[12][12]) * math.pi / 180
    deg_b_z = float(data[12][13]) * math.pi / 180
"""
Race irregular geometry
"""
k_ip_type_o = int(data[12][14])
# Arbitrary user supplied outer race inertial parameters
if k_ip_type_o <= 0:
    m_o = den_o * math.pi * W_o * (D_o_u ** 2 / 4 - D_o_d ** 2 / 4)
    # Outer race mass
    I_o_x = m_o * (D_o_u ** 2 + D_o_d ** 2) / 8
    # Interia of outer race in dir x
    I_o_y = I_o_x / 2 + m_o * W_o ** 2 / 12
    # Interia of outer race in dir y
    I_o_z = I_o_x / 2 + m_o * W_o ** 2 / 12
    # Interia of outer race in dir z
    geo_cen_o_x = 0.
    # Outer race geometric center in dir x
    geo_cen_o_y = 0.
    # Outer race geometric center in dir y
    geo_cen_o_z = 0.
    # Outer race geometric center in dir z
    deg_o_x = 0 * math.pi / 180
    # Outer race geo center to principal triad in dir x
    deg_o_y = 0 * math.pi / 180
    # Outer race geo center to principal triad in dir y
    deg_o_z = 0 * math.pi / 180
    # Outer race geo center to principal triad in dir z
else:
    m_o = float(data[12][15])
    I_o_x = float(data[12][16])
    I_o_y = float(data[12][17])
    I_o_z = float(data[12][18])
    geo_cen_o_x = float(data[12][19])
    geo_cen_o_y = float(data[12][20])
    geo_cen_o_z = float(data[12][21])
    deg_o_x = float(data[12][22]) * math.pi / 180
    deg_o_y = float(data[12][23]) * math.pi / 180
    deg_o_z = float(data[12][24]) * math.pi / 180

k_geo_imc_type_o = int(data[12][25])
# Imperfections in outer race geometry
if k_geo_imc_type_o <= 0:
    var_o_r0 = 0.
    # Deviation of the semi-major axis from nominal or radius
    var_o_r1 = 0.
    # Deviation of the semi-minor axis from nominal or radius
    var_o_r2 = 0 * math.pi / 180
    # Orientation (deg) of major axis relative in dir x
elif k_geo_imc_type_o == 1:
    var_o_r0 = float(data[12][26])
    var_o_r1 = float(data[12][27])
    var_o_r2 = float(data[12][28]) * math.pi / 180
else:
    var_o_r0 = float(data[12][26])
    var_o_r1 = float(data[12][27])
    var_o_r2 = float(data[12][28]) * math.pi / 180

k_ip_type_i = int(data[12][29])
# Arbitrary user supplied inner race inertial parameters
if k_ip_type_i <= 0:
    m_i = den_i * math.pi * W_i * (D_i_u ** 2 / 4 - D_i_d ** 2 / 4)
    # Inner race mass
    I_i_x = m_i * (D_i_u ** 2 + D_i_d ** 2) / 8
    # Interia of inner race in dir x
    I_i_y = I_i_x / 2 + m_i * W_i ** 2 / 12
    # Interia of inner race in dir y
    I_i_z = I_i_x / 2 + m_i * W_i ** 2 / 12
    # Interia of inner race in dir z
    geo_cen_i_x = 0.
    # Inner race geometric center in dir x
    geo_cen_i_y = 0.
    # Inner race geometric center in dir y
    geo_cen_i_z = 0.
    # Inner race geometric center in dir z
    deg_i_x = 0 * math.pi / 180
    # Inner race geo center to principal triad in dir x
    deg_i_y = 0 * math.pi / 180
    # Inner race geo center to principal triad in dir y
    deg_i_z = 0 * math.pi / 180
    # Inner race geo center to principal triad in dir z
else:
    m_i = float(data[12][30])
    I_i_x = float(data[12][31])
    I_i_y = float(data[12][32])
    I_i_z = float(data[12][33])
    geo_cen_i_x = float(data[12][34])
    geo_cen_i_y = float(data[12][35])
    geo_cen_i_z = float(data[12][36])
    deg_i_x = float(data[12][37]) * math.pi / 180
    deg_i_y = float(data[12][38]) * math.pi / 180
    deg_i_z = float(data[12][39]) * math.pi / 180

k_geo_imc_type_i = int(data[12][40])
# Imperfections in outer race geometry
if k_geo_imc_type_i <= 0:
    var_i_r0 = 0.
    # Deviation of the semi-major axis from nominal ir radius
    var_i_r1 = 0.
    # Deviation of the semi-minor axis from nominal ir radius
    var_i_r2 = 0 * math.pi / 180
    # Orientation (deg) of major axis relative in dir x
elif k_geo_imc_type_i == 1:
    var_i_r0 = float(data[12][41])
    var_i_r1 = float(data[12][42])
    var_i_r2 = float(data[12][43]) * math.pi / 180
else:
    var_i_r0 = float(data[12][41])
    var_i_r1 = float(data[12][42])
    var_i_r2 = float(data[12][43]) * math.pi / 180
"""
Cage irregular geometry
"""
k_ip_type_c = int(data[12][44])
# Arbitrary user supplied cage inertial parameters
if k_ip_type_c <= 0:
    m_c = den_c * (math.pi * W_c * (D_c_u ** 2 - D_c_d ** 2) / 4 -
                   math.pi * (D_c_u - D_c_d) / 8 * D_p_u ** 2 * n)
    # Cage mass
    I_c_x = m_c * (D_c_u ** 2 + D_c_d ** 2) / 8
    # Interia of cage in dir x
    I_c_y = I_c_x / 2 + m_c * W_c ** 2 / 12
    # Interia of cage in dir y
    I_c_z = I_c_x / 2 + m_c * W_c ** 2 / 12
    # Interia of cage in dir z
    geo_cen_c_x = 0.
    # Cage geometric center in dir x
    geo_cen_c_y = 0.
    # Cage geometric center in dir y
    geo_cen_c_z = 0.
    # Cage geometric center in dir z
    deg_c_x = 0 * math.pi / 180
    # Cage geo center to principal triad in dir x
    deg_c_y = 0 * math.pi / 180
    # Cage geo center to principal triad in dir y
    deg_c_z = 0 * math.pi / 180
    # Cage geo center to principal triad in dir z
else:
    m_c = float(data[12][45])
    I_c_x = float(data[12][46])
    I_c_y = float(data[12][47])
    I_c_z = float(data[12][48])
    geo_cen_c_x = float(data[12][49])
    geo_cen_c_y = float(data[12][50])
    geo_cen_c_z = float(data[12][51])
    deg_c_x = float(data[12][52]) * math.pi / 180
    deg_c_y = float(data[12][53]) * math.pi / 180
    deg_c_z = float(data[12][54]) * math.pi / 180

k_geo_imc_type_c = int(data[12][55])
# Imperfections in outer race geometry
if k_geo_imc_type_c <= 0:
    var_c_r0 = 0.
    # Deviation of the semi-major axis from nominal ir radius
    var_c_r1 = 0.
    # Deviation of the semi-minor axis from nominal ir radius
    var_c_r2 = 0 * math.pi / 180
    # Orientation (deg) of major axis relative in dir x
elif k_geo_imc_type_c == 1:
    var_c_r0 = float(data[12][56])
    var_c_r1 = float(data[12][57])
    var_c_r2 = float(data[12][58]) * math.pi / 180
else:
    var_c_r0 = float(data[12][56])
    var_c_r1 = float(data[12][57])
    var_c_r2 = float(data[12][58]) * math.pi / 180
"""
Cage pocket irregular geometry
"""
k_ip_type_p = int(data[12][59])
# Imperfections in cage pockets
if k_ip_type_p <= 0:
    var_num_p = 0
    # Number of the pocket center imperfections
    var_inr_p = 1
    # Interval of the pocket center imperfections (must >= 1)
    var_p_x = 0.
    # Axial position of the pocket center to the cage center
    var_p_r = 0.
    # Ridial position of the pocket center to the cage center
    var_num_p_ang = 0
    # Number of the angle imperfections
    var_inr_p_ang = 1
    # Interval of the angle imperfections (must >= 1)
    var_p_ang_x = 0 * math.pi / 180
    # Variation in transformation angle thetax
    var_p_ang_y = 0 * math.pi / 180
    # Variation in transformation angle thetay
    var_p_ang_z = 0 * math.pi / 180
    # Variation in transformation angle thetaz
else:
    var_num_p = int(data[12][60])
    var_inr_p = int(data[12][61])
    var_p_x = float(data[12][62])
    var_p_r = float(data[12][63])
    var_num_p_ang = int(data[12][64])
    var_inr_p_ang = int(data[12][65])
    var_p_ang_x = float(data[12][66]) * math.pi / 180
    var_p_ang_y = float(data[12][67]) * math.pi / 180
    var_p_ang_z = float(data[12][68]) * math.pi / 180
###############################################################################
###############################################################################
#                        Time step and condition data                         #
###############################################################################
"""
Solve parameters
"""
sol_type = int(data[13][1])
# 0 for start, 1 for restart
run_type = int(data[13][2])
# Integration algorithm"""
start_time = float(data[13][3])
# Start time of this run
end_time = float(data[13][4])
# Final value of dimensionless time
time_step_num = int(data[13][5])
# Maximum number of steps for this run, always suggest over 10000 steps for
# real time equaling to 1.0s
plot_step_freq = int(data[13][6])
# Plot frequecy of steps for this run
"""
Conditions
"""
F_x = float(data[13][7])
# Force to x direction
F_y = float(data[13][8])
# Force to y direction
F_z = float(data[13][9])
# Force to z direction
M_y = float(data[13][10])
# Moment to y direction
if M_y != 0:
    mis_o_y, mis_i_y = 0., 0.
    # Reset misalignment of y, both outer race and inner race
M_z = float(data[13][11])
# Moment to z direction
if M_z != 0:
    mis_o_z, mis_i_z = 0., 0.
    # Reset misalignment of z, both outer race and inner race
k_usdf_type = int(data[13][12])
# Type of user defined force
"""
Parameters process
"""
F_max = max(abs(F_x), abs(F_y), abs(F_z))
end_time = end_time * math.sqrt(F_max / (m_b * R_b))
###############################################################################
###############################################################################
#                             Save all base data                              #
###############################################################################
base_data_name = [
    'base_hd',
    'base_bd',
    'base_obd',
    'base_ord',
    'base_ird',
    'base_cd',
    'base_od',
    'base_hasd',
    'base_mfd',
    'base_gd',
    'base_bld',
    'base_igd',
    'base_tsacd'
]
# Base data name
base_hd = {
    'ini_temp_r':ini_temp_r,
    'ini_temp_h':ini_temp_h,
    'ini_temp_s':ini_temp_s,
    'ini_temp_o':ini_temp_o,
    'ini_temp_i':ini_temp_i,
    'ini_temp_b':ini_temp_b,
    'ini_temp_c':ini_temp_c
}
# Base data of heat data
base_bd = {
    'D_b':D_b,
    'R_b':R_b,
    'n':n,
    'mat_type_b':mat_type_b,
    'mat_prop_b':mat_prop_b,
    'den_b':den_b,
    'E_b':E_b,
    'po_b':po_b,
    'elas_stra_limt_b':elas_stra_limt_b,
    'coeff_ther_exp_b':coeff_ther_exp_b,
    'ther_cond_b':ther_cond_b,
    'spec_heat_b':spec_heat_b,
    'wear_coff_b':wear_coff_b,
    'hard_coff_b':hard_coff_b,
    'von_mises_stress_b':von_mises_stress_b,
    'mat_fac_type_b':mat_fac_type_b,
    'proc_fac_type_b':proc_fac_type_b,
    'rms_b':rms_b,
}
# Base data of ball data
base_obd = {
    'D_m':D_m,
    'R_m':R_m,
    'free_con_ang':free_con_ang
}
# Base data of other bearing data
base_ord = {
    'D_o_u':D_o_u,
    'R_o_u':R_o_u,
    'D_o_m':D_o_m,
    'R_o_m':R_o_m,
    'f_o':f_o,
    'D_o_d':D_o_d,
    'R_o_d':R_o_d,
    'W_o':W_o,
    'Shim_thknss_o':Shim_thknss_o,
    'sam_o':sam_o,
    'cam_o':cam_o,
    'shim_ang_o':shim_ang_o,
    'mat_type_o':mat_type_o,
    'mat_prop_o':mat_prop_o,
    'den_o':den_o,
    'E_o':E_o,
    'po_o':po_o,
    'elas_stra_limt_o':elas_stra_limt_o,
    'coeff_ther_exp_o':coeff_ther_exp_o,
    'ther_cond_o':ther_cond_o,
    'spec_heat_o':spec_heat_o,
    'wear_coff_o':wear_coff_o,
    'hard_coff_o':hard_coff_o,
    'von_mises_stress_o':von_mises_stress_o,
    'mat_fac_type_o':mat_fac_type_o,
    'proc_fac_type_o':proc_fac_type_o,
    'rms_o':rms_o,
    'dmpg_b_o':dmpg_b_o,
    'rpm_o':rpm_o,
    'mis_o_y':mis_o_y,
    'mis_o_z':mis_o_z,
    'k_tra_o_x':k_tra_o_x,
    'k_tra_o_y':k_tra_o_y,
    'k_tra_o_z':k_tra_o_z,
    'k_rot_o_x':k_rot_o_x,
    'k_rot_o_y':k_rot_o_y,
    'k_rot_o_z':k_rot_o_z,
    'sub_f_o_x':sub_f_o_x,
    'sub_f_o_y':sub_f_o_y,
    'sub_f_o_z':sub_f_o_z,
    'sub_m_o_x':sub_m_o_x,
    'sub_m_o_y':sub_m_o_y,
    'sub_m_o_z':sub_m_o_z
}
# Base data of outer race data
base_ird = {
    'D_i_d':D_i_d,
    'R_i_d':R_i_d,
    'D_i_m':D_i_m,
    'R_i_m':R_i_m,
    'f_i':f_i,
    'D_i_u':D_i_u,
    'R_i_u':R_i_u,
    'W_i':W_i,
    'Shim_thknss_i':Shim_thknss_i,
    'sam_i':sam_i,
    'cam_i':cam_i,
    'shim_ang_i':shim_ang_i,
    'mat_type_i':mat_type_i,
    'mat_prop_i':mat_prop_i,
    'den_i':den_i,
    'E_i':E_i,
    'po_i':po_i,
    'elas_stra_limt_i':elas_stra_limt_i,
    'coeff_ther_exp_i':coeff_ther_exp_i,
    'ther_cond_i':ther_cond_i,
    'spec_heat_i':spec_heat_i,
    'wear_coff_i':wear_coff_i,
    'hard_coff_i':hard_coff_i,
    'von_mises_stress_i':von_mises_stress_i,
    'mat_fac_type_i':mat_fac_type_i,
    'proc_fac_type_i':proc_fac_type_i,
    'rms_i':rms_i,
    'dmpg_b_i':dmpg_b_i,
    'rpm_i':rpm_i,
    'mis_i_y':mis_i_y,
    'mis_i_z':mis_i_z,
    'k_tra_i_x':k_tra_i_x,
    'k_tra_i_y':k_tra_i_y,
    'k_tra_i_z':k_tra_i_z,
    'k_rot_i_x':k_rot_i_x,
    'k_rot_i_y':k_rot_i_y,
    'k_rot_i_z':k_rot_i_z,
    'sub_f_i_x':sub_f_i_x,
    'sub_f_i_y':sub_f_i_y,
    'sub_f_i_z':sub_f_i_z,
    'sub_m_i_x':sub_m_i_x,
    'sub_m_i_y':sub_m_i_y,
    'sub_m_i_z':sub_m_i_z
}
# Base data of inner race data
base_cd = {
    'n_cseg':n_cseg,
    'D_c_u':D_c_u,
    'R_c_u':R_c_u,
    'D_c_d':D_c_d,
    'R_c_d':R_c_d,
    'D_c_g':D_c_g,
    'R_c_g':R_c_g,
    'D_p_u':D_p_u,
    'R_p_u':R_p_u,
    'D_p_d':D_p_d,
    'R_p_d':R_p_d,
    'W_c':W_c,
    'Sg':Sg,
    'Cg_u':Cg_u,
    'Cg_d':Cg_d,
    'Bg_l':Bg_l,
    'Bg_r':Bg_r,
    'mat_type_c':mat_type_c,
    'mat_prop_c':mat_prop_c,
    'den_c':den_c,
    'E_c':E_c,
    'po_c':po_c,
    'elas_stra_limt_c':elas_stra_limt_c,
    'coeff_ther_exp_c':coeff_ther_exp_c,
    'ther_cond_c':ther_cond_c,
    'spec_heat_c':spec_heat_c,
    'wear_coff_c':wear_coff_c,
    'hard_coff_c':hard_coff_c,
    'von_mises_stress_c':von_mises_stress_c,
    'rms_c':rms_c,
    'dmpg_b_c':dmpg_b_c,
    'dmpg_c_r':dmpg_c_r,
    'poc_lub_film':poc_lub_film,
    'poc_type':poc_type,
    'Poc_cls':Poc_cls,
    'cage_freedom':cage_freedom,
    'c_r_guide_type':c_r_guide_type,
    'int_sol':int_sol,
    'wv_ratio':wv_ratio,
    'av_ratio':av_ratio,
    'cage_mass_cen_x':cage_mass_cen_x,
    'cage_mass_cen_y':cage_mass_cen_y,
    'cage_mass_cen_z':cage_mass_cen_z
}
# Base data of cage data
base_od = {
    'oil_type':oil_type,
    'oil_prop':oil_prop,
    'trac_alpha':trac_alpha,
    'trac_beta':trac_beta,
    'trac_vis':trac_vis,
    'vis_lub':vis_lub,
    'den_lub':den_lub,
    'spec_heat_lub':spec_heat_lub,
    'ther_cond_lub':ther_cond_lub,
    'dvis_lub':dvis_lub,
    'vis_coeff_0':vis_coeff_0,
    'vis_coeff_1':vis_coeff_1,
    'b_r_lub_type':b_r_lub_type,
    'k_b_r_trac_type':k_b_r_trac_type,
    'kai0_0':kai0_0,
    'kaim_0':kaim_0,
    'kaiin_0':kaiin_0,
    'um_0':um_0,
    'b_c_lub_type':b_c_lub_type,
    'k_b_c_trac_type':k_b_c_trac_type,
    'f_b_c':f_b_c,
    'c_r_lub_type':c_r_lub_type,
    'k_c_r_trac_type':k_c_r_trac_type,
    'f_c_r':f_c_r,
    'k_b_b_trac_type':k_b_b_trac_type,
    'b_b_limt_film':b_b_limt_film,
    'f_b_b':f_b_b,
    'str_parm':str_parm,
    'b_o_limt_film':b_o_limt_film,
    'b_i_limt_film':b_i_limt_film,
    'b_c_limt_film':b_c_limt_film,
    'c_r_limt_film':c_r_limt_film,
    'k_chrn_type':k_chrn_type,
    'den_ratio':den_ratio,
    'vis_ratio':vis_ratio,
    'chrn_pres':chrn_pres,
    'ini_temp_chrn':ini_temp_chrn
}
# Base data of oil data
base_hasd = {
    'D_h':D_h,
    'D_s':D_s,
    'Fit_o_h':Fit_o_h,
    'Fit_i_s':Fit_i_s,
    'mat_type_h':mat_type_h,
    'mat_prop_h':mat_prop_h,
    'den_h':den_h,
    'E_h':E_h,
    'po_h':po_h,
    'elas_stra_limt_h':elas_stra_limt_h,
    'coeff_ther_exp_h':coeff_ther_exp_h,
    'ther_cond_h':ther_cond_h,
    'spec_heat_h':spec_heat_h,
    'wear_coff_h':wear_coff_h,
    'hard_coff_h':hard_coff_h,
    'von_mises_stress_h':von_mises_stress_h,
    'mat_type_s':mat_type_s,
    'mat_prop_s':mat_prop_s,
    'den_s':den_s,
    'E_s':E_s,
    'po_s':po_s,
    'elas_stra_limt_s':elas_stra_limt_s,
    'coeff_ther_exp_s':coeff_ther_exp_s,
    'ther_cond_s':ther_cond_s,
    'spec_heat_s':spec_heat_s,
    'wear_coff_s':wear_coff_s,
    'hard_coff_s':hard_coff_s,
    'von_mises_stress_s':von_mises_stress_s
}
# Base data of house and shaft data
base_mfd = {
    'k_brg_mov':k_brg_mov,
    'R_brg_orb':R_brg_orb,
    'ini_brg_ang_pos':ini_brg_ang_pos,
    'brg_mov_dir':brg_mov_dir,
    'brg_ang_vel':brg_ang_vel,
    'brg_load_frac_o':brg_load_frac_o,
    'brg_load_frac_i':brg_load_frac_i
}
# Base data of moving frame
base_gd = {
    'g':g,
    'G_vec_0':G_vec_0,
    'G_vec_1':G_vec_1,
    'G_vec_2':G_vec_2
}
# Base data of gravity
base_bld = {
    'k_life_freq':k_life_freq,
    'k_life_cons':k_life_cons,
    'res_stress_o':res_stress_o,
    'res_stress_i':res_stress_i,
    'str_limt_fac':str_limt_fac
}
# Base data of bearing life
base_igd = {
    'k_ip_type_b':k_ip_type_b,
    'var_num_b':var_num_b,
    'var_inr_b':var_inr_b,
    'm_b':m_b,
    'I_b_x':I_b_x,
    'I_b_y':I_b_y,
    'I_b_z':I_b_z,
    'geo_cen_b_x':geo_cen_b_x,
    'geo_cen_b_y':geo_cen_b_y,
    'geo_cen_b_z':geo_cen_b_z,
    'deg_b_x':deg_b_x,
    'deg_b_y':deg_b_y,
    'deg_b_z':deg_b_z,
    'k_ip_type_o':k_ip_type_o,
    'm_o':m_o,
    'I_o_x':I_o_x,
    'I_o_y':I_o_y,
    'I_o_z':I_o_z,
    'geo_cen_o_x':geo_cen_o_x,
    'geo_cen_o_y':geo_cen_o_y,
    'geo_cen_o_z':geo_cen_o_z,
    'deg_o_x':deg_o_x,
    'deg_o_y':deg_o_y,
    'deg_o_z':deg_o_z,
    'k_geo_imc_type_o':k_geo_imc_type_o,
    'var_o_r0':var_o_r0,
    'var_o_r1':var_o_r1,
    'var_o_r2':var_o_r2,
    'k_ip_type_i':k_ip_type_i,
    'm_i':m_i,
    'I_i_x':I_i_x,
    'I_i_y':I_i_y,
    'I_i_z':I_i_z,
    'geo_cen_i_x':geo_cen_i_x,
    'geo_cen_i_y':geo_cen_i_y,
    'geo_cen_i_z':geo_cen_i_z,
    'deg_i_x':deg_i_x,
    'deg_i_y':deg_i_y,
    'deg_i_z':deg_i_z,
    'k_geo_imc_type_i':k_geo_imc_type_i,
    'var_i_r0':var_i_r0,
    'var_i_r1':var_i_r1,
    'var_i_r2':var_i_r2,
    'k_geo_imc_type_c':k_geo_imc_type_c,
    'var_c_r0':var_c_r0,
    'var_c_r1':var_c_r1,
    'var_c_r2':var_c_r2,
    'k_ip_type_c':k_ip_type_c,
    'm_c':m_c,
    'I_c_x':I_c_x,
    'I_c_y':I_c_y,
    'I_c_z':I_c_z,
    'geo_cen_c_x':geo_cen_c_x,
    'geo_cen_c_y':geo_cen_c_y,
    'geo_cen_c_z':geo_cen_c_z,
    'deg_c_x':deg_c_x,
    'deg_c_y':deg_c_y,
    'deg_c_z':deg_c_z,
    'k_ip_type_p':k_ip_type_p,
    'var_num_p':var_num_p,
    'var_inr_p':var_inr_p,
    'var_p_x':var_p_x,
    'var_p_r':var_p_r,
    'var_num_p_ang':var_num_p_ang,
    'var_inr_p_ang':var_inr_p_ang,
    'var_p_ang_x':var_p_ang_x,
    'var_p_ang_y':var_p_ang_y,
    'var_p_ang_z':var_p_ang_z
}
# Base data of irregular geometry
base_tsacd = {
    'sol_type':sol_type,
    'run_type':run_type,
    'start_time':start_time,
    'end_time':end_time,
    'time_step_num':time_step_num,
    'plot_step_freq':plot_step_freq,
    'F_x':F_x,
    'F_y':F_y,
    'F_z':F_z,
    'M_y':M_y,
    'M_z':M_z,
    'k_usdf_type':k_usdf_type
}
# Base data of time steps and condition
base_data = [
    base_data_name,
    base_hd,
    base_bd,
    base_obd,
    base_ord,
    base_ird,
    base_cd,
    base_od,
    base_hasd,
    base_mfd,
    base_gd,
    base_bld,
    base_igd,
    base_tsacd
]
# Base data
mod_name = [
    'mod_tc',
    'mod_es',
    'mod_nlp',
    'mod_ip',
    'mod_cf',
    'mod_brcs',
    'mod_brcs_',
    'mod_brcf',
    'mod_brcf_',
    'mod_brtf',
    'mod_brtf_',
    'mod_bcf',
    'mod_crf',
    'mod_mff',
    'mod_gf',
    'mod_df',
    'mod_bbf',
    'mod_orhf',
    'mod_udfc',
    'mod_udf',
    'mod_flc',
    'mod_fl',
    'mod_pl',
    'mod_so'
]
# Module index of list name
mod_index_name = [
    'mod_tc_index',
    'mod_es_index',
    'mod_nlp_index',
    'mod_ip_index',
    'mod_cf_index',
    'mod_brcs_index',
    'mod_brcs_index_',
    'mod_brcf_index',
    'mod_brcf_index_',
    'mod_brtf_index',
    'mod_brtf_index_',
    'mod_bcf_index',
    'mod_crf_index',
    'mod_mff_index',
    'mod_gf_index',
    'mod_df_index',
    'mod_bbf_index',
    'mod_orhf_index',
    'mod_udfc_index',
    'mod_udf_index',
    'mod_flc_index',
    'mod_fl_index',
    'mod_pl_index',
    'mod_so_index'
]
# Module index of list name
mod_tc_index = [
    'ini_temp_o',
    'ini_temp_i',
    'ini_temp_h',
    'ini_temp_s',
    'ini_temp_c',
    'ini_temp_r',
    'ini_temp_chrn'
]
# Module index of temperature_change
mod_es_index = [
    'D_p_u',
    'R_o_m',
    'Fit_i_s',
    'Fit_o_h',
    'n',
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
    'sto10',
    'sto12',
    'sti01',
    'sti02',
    'sti11',
    'sti12',
    'stc12'
]
# Module index of expansion_subcall
mod_nlp_index = [
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
]
# Module index of no_load_position
mod_ip_index = [
    'F_x',
    'F_y',
    'F_z',
    'R_b',
    'f_i',
    'f_o',
    'free_con_ang',
    'n',
    'rpm_i',
    'rpm_o'
]
# Module index of initial_position
mod_cf_index = [
    'D_b',
    'I_b_z',
    'R_b',
    'ee_b_o',
    'ee_b_i',
    'f_i',
    'f_o',
    'm_b',
    'n',
    'rpm_i',
    'rpm_o'
]
# Module index of ball_centrifugal_forece
mod_brcs_index = [
    'D_b',
    'D_m',
    'T_I_imis',
    'T_I_omis',
    'T_bp_b',
    'Shim_thknss_i',
    'Shim_thknss_o',
    'f_i',
    'f_o',
    'free_con_ang',
    'k_geo_imc_type_i',
    'k_geo_imc_type_o',
    'r_bg_bm_b',
    'n',
    'var_i_r0',
    'var_i_r1',
    'var_i_r2',
    'var_o_r0',
    'var_o_r1',
    'var_o_r2'
]
# Module index of ball_race_contact_strain
mod_brcs_index_ = [
    'D_b',
    'Shim_thknss_i',
    'Shim_thknss_o',
    'f_i',
    'f_o',
    'max_rs',
    'n'
]
# Module index of _ball_race_contact_strain
mod_brcf_index = [
    'E_b_i',
    'E_b_o',
    'K_b_i',
    'K_b_o',
    'R_b',
    'R_b_i',
    'R_b_o',
    'de_b_i',
    'de_b_o',
    'ke_b_i',
    'ke_b_o',
    'n'
]
# Module index of ball_race_contact_force
mod_brcf_index_ = [
    'E_b_i',
    'E_b_o',
    'K_b_i',
    'K_b_o',
    'R_b',
    'R_b_i',
    'R_b_o',
    'Shim_thknss_i',
    'Shim_thknss_o',
    'de_b_i',
    'de_b_o',
    'ke_b_i',
    'ke_b_o',
    'n'
]
# Module index of _ball_race_contact_force
mod_brtf_index = [
    'A_0',
    'B_0',
    'C_0',
    'D_0',
    'D_b',
    'R_b',
    'R_yipu_b_o',
    'R_yipu_b_i',
    'b_i_limt_film',
    'b_o_limt_film',
    'b_r_lub_type',
    'dmpg_b_i',
    'dmpg_b_o',
    'ep_b_i',
    'ep_b_o',
    'f_i',
    'f_o',
    'hj',
    'k_b_r_trac_type',
    'm_b',
    'm_i',
    'm_o',
    'n',
    'oil_type',
    'r_bg_bm_b',
    'str_parm',
    'tj'
]
# Module index of ball_race_traction_force
mod_brtf_index_ = [
    'A_0',
    'B_0',
    'C_0',
    'D_0',
    'D_b',
    'R_b',
    'R_yipu_b_i',
    'R_yipu_b_o',
    'Shim_thknss_i',
    'Shim_thknss_o',
    'b_i_limt_film',
    'b_o_limt_film',
    'b_r_lub_type',
    'dmpg_b_i',
    'dmpg_b_o',
    'ep_b_i',
    'ep_b_o',
    'f_i',
    'f_o',
    'hj',
    'k_b_r_trac_type',
    'm_b',
    'm_i',
    'm_o',
    'n',
    'oil_type',
    'r_bg_bm_b',
    'str_parm',
    'tj'
]
# Module index of _ball_race_traction_force
mod_bcf_index = [
    'Bg_l',
    'Bg_r',
    'D_b',
    'D_c_d',
    'D_c_u',
    'D_p_u',
    'E_b_c',
    'E_c',
    'K_b_c',
    'K_b_c_',
    'Poc_cls',
    'R_b',
    'R_b_c',
    'R_p_d',
    'R_p_u',
    'T_cp_c',
    'T_p_pvar',
    'W_c',
    'b_c_limt_film',
    'b_c_lub_type',
    'de_b_c',
    'dmpg_b_c',
    'elas_stra_limt_c',
    'f_b_c',
    'hj8',
    'int_sol',
    'k_b_c_trac_type',
    'k_con_type_b_p',
    'ke_b_c',
    'm_b',
    'm_c',
    'n',
    'num_gs',
    'oil_type',
    'phi_pvar',
    'poc_lub_film',
    'poc_type',
    'r_bg_bm_b',
    'r_cg_cm_c',
    'r_pvar_cg_c',
    't_b_c',
    'tj8',
    'w0',
    'w1',
    'w2',
    'w3',
    'xh'
]
# Module index of ball_cage_force
mod_crf_index = [
    'Bg_l',
    'Bg_r',
    'Cg_d',
    'Cg_u',
    'E_c',
    'E_i',
    'E_o',
    'R_c_g',
    'Sg',
    'c_r_guide_type',
    'c_r_limt_film',
    'c_r_lub_type',
    'dmpg_c_r',
    'elas_stra_limt_c',
    'f_c_r',
    'k_c_r_trac_type',
    'm_c',
    'm_i',
    'm_o',
    'oil_type',
    'po_c',
    'po_i',
    'po_o',
    'r_cg_cm_c'
]
# Module index of cage_race_force
mod_mff_index = [
    'R_brg_orb',
    'brg_ang_vel',
    'brg_load_frac_i',
    'brg_load_frac_o',
    'brg_mov_dir',
    'm_b',
    'm_c',
    'm_i',
    'm_o',
    'n',
    'n_cseg'
]
# Module index of moving_frame_force
mod_gf_index = [
    'G_vec_0',
    'G_vec_1',
    'G_vec_2',
    'brg_ang_vel',
    'brg_mov_dir',
    'ini_brg_ang_pos',
    'g',
    'k_brg_mov',
    'm_b',
    'm_c',
    'm_i',
    'm_o',
    'n',
    'n_cseg'
]
# Module index of gravity_force
mod_df_index = [
    'Cg_d',
    'Cg_u',
    'D_b',
    'D_c_d',
    'D_c_u',
    'D_m',
    'R_m',
    'R_p_u',
    'W_c',
    'chrn_pres',
    'den_ratio',
    'n',
    'n_cseg',
    'oil_type',
    'vis_ratio'
]
# Module index of drag_force
mod_bbf_index = [
    'D_b',
    'E_b',
    'R_b',
    'b_b_trac_type',
    'b_b_limt_film',
    'elas_stra_limt_b',
    'f_b_b',
    'n',
    'po_b'
]
# Module index of ball_ball_force
mod_orhf_index = [
    'R_o_u',
    'dmpg_o_h',
    'k_ok_h',
    'n'
]
# Module index of outer_race_house_force
mod_udfc_index = [
    'end_time',
    'k_usdf_type',
    's_time',
    'time_step_num'
]
# Module index of user_defined_force_curve
mod_udf_index = [
    'end_time',
    'time_step_num'
]
# Module index of user_defined_force
mod_flc_index = [
    'D_b',
    'D_m',
    'E_b',
    'E_i',
    'E_o',
    'Shim_thknss_i',
    'Shim_thknss_o',
    'brg_type',
    'f_i',
    'f_o',
    'hard_coff_b',
    'hard_coff_i',
    'hard_coff_o',
    'k_life_cons',
    'mat_fac_type_b',
    'mat_fac_type_i',
    'mat_fac_type_o',
    'mat_type_b',
    'max_rs',
    'n',
    'po_b',
    'po_i',
    'po_o',
    'proc_fac_type_b',
    'proc_fac_type_i',
    'proc_fac_type_o',
    'res_stress_i',
    'res_stress_o',
    'rms_b',
    'rms_i',
    'rms_o'
]
# Module index of fatigue_life_constant
mod_fl_index = [
    'D_b',
    'D_m',
    'R_b',
    'a_b_i',
    'a_b_o',
    'b_b_i',
    'b_b_o',
    'f_i',
    'f_o',
    'max_rs',
    'n',
    'str_limt_fac',
    'von_mises_stress_b',
    'von_mises_stress_i',
    'von_mises_stress_o'
]
# Module index of fatigue_life
mod_pl_index = [
    'n',
    'n_cseg'
]
# Module index of power_loss
mod_so_index = [
    'E_b_c',
    'E_b_i',
    'E_b_o',
    'K_b_c',
    'K_b_i',
    'K_b_o',
    'K_b_c_',
    'Shim_thknss_i',
    'Shim_thknss_o',
    'k_brg_mov',
    'k_chrn_type',
    'k_life_freq',
    'n_cseg'
]
# Module index of store_output
mod_index_data = [
    mod_name,
    mod_index_name,
    mod_tc_index,
    mod_es_index,
    mod_nlp_index,
    mod_ip_index,
    mod_cf_index,
    mod_brcs_index,
    mod_brcs_index_,
    mod_brcf_index,
    mod_brcf_index_,
    mod_brtf_index,
    mod_brtf_index_,
    mod_bcf_index,
    mod_crf_index,
    mod_mff_index,
    mod_gf_index,
    mod_df_index,
    mod_bbf_index,
    mod_orhf_index,
    mod_udfc_index,
    mod_udf_index,
    mod_flc_index,
    mod_fl_index,
    mod_pl_index,
    mod_so_index
]
# Module index data
(mod_tc,
 mod_es,
 mod_nlp,
 mod_ip,
 mod_cf,
 mod_brcs,
 mod_brcs_,
 mod_brcf,
 mod_brcf_,
 mod_brtf,
 mod_brtf_,
 mod_bcf,
 mod_crf,
 mod_mff,
 mod_gf,
 mod_df,
 mod_bbf,
 mod_orhf,
 mod_udfc,
 mod_udf,
 mod_flc,
 mod_fl,
 mod_pl,
 mod_so
 ) = (
     (),
     (),
     (),
     (),
     (),
     (),
     (),
     (),
     (),
     (),
     (),
     (),
     (),
     (),
     (),
     (),
     (),
     (),
     (),
     (),
     (),
     (),
     (),
     ()
     )
for i in range(1):
    with open('Input.pickle', 'wb') as f:
        pickle.dump([base_data, mod_index_data], f)
###############################################################################
#                              Some matrix build                              #
###############################################################################
if brg_type < 10:
    plot_step_num = int(time_step_num / plot_step_freq)
    ###########################################################################
    #                            Base output array                            #
    ###########################################################################
    aa = np.zeros((37+12*n, plot_step_num))
    ###########################################################################
    #            Contact(Yaw) angle, force between ball and race              #
    ###########################################################################
    con_ang_b_o_0 = np.zeros((plot_step_num, n, 1, 1))
    con_ang_b_o_1 = np.zeros((plot_step_num, n, 1, 1))
    con_ang_b_i_0 = np.zeros((plot_step_num, n, 1, 1))
    con_ang_b_i_1 = np.zeros((plot_step_num, n, 1, 1))
    con_half_len_b_o = np.zeros((plot_step_num, n, 1, 1))
    con_half_len_b_i = np.zeros((plot_step_num, n, 1, 1))
    con_half_wid_b_o = np.zeros((plot_step_num, n, 1, 1))
    con_half_wid_b_i = np.zeros((plot_step_num, n, 1, 1))
    con_pre_b_o = np.zeros((plot_step_num, n, 1, 1))
    con_pre_b_i = np.zeros((plot_step_num, n, 1, 1))
    con_load_dis_b_o = np.zeros((plot_step_num, n, 1, 1))
    con_load_dis_b_i = np.zeros((plot_step_num, n, 1, 1))
    con_for_b_o = np.zeros((plot_step_num, n, 1, 1))
    con_for_b_i = np.zeros((plot_step_num, n, 1, 1))
    if Shim_thknss_o > 0 or Shim_thknss_i > 0:
        con_nor_dis_b_o_ = np.zeros((plot_step_num, n, 1, 1))
        con_nor_dis_b_i_ = np.zeros((plot_step_num, n, 1, 1))
        con_ang_b_o_0_ = np.zeros((plot_step_num, n, 1, 1))
        con_ang_b_o_1_ = np.zeros((plot_step_num, n, 1, 1))
        con_ang_b_i_0_ = np.zeros((plot_step_num, n, 1, 1))
        con_ang_b_i_1_ = np.zeros((plot_step_num, n, 1, 1))
        con_half_len_b_o_ = np.zeros((plot_step_num, n, 1, 1))
        con_half_len_b_i_ = np.zeros((plot_step_num, n, 1, 1))
        con_half_wid_b_o_ = np.zeros((plot_step_num, n, 1, 1))
        con_half_wid_b_i_ = np.zeros((plot_step_num, n, 1, 1))
        con_pre_b_o_ = np.zeros((plot_step_num, n, 1, 1))
        con_pre_b_i_ = np.zeros((plot_step_num, n, 1, 1))
        con_load_dis_b_o_ = np.zeros((plot_step_num, n, 1, 1))
        con_load_dis_b_i_ = np.zeros((plot_step_num, n, 1, 1))
        con_for_b_o_ = np.zeros((plot_step_num, n, 1, 1))
        con_for_b_i_ = np.zeros((plot_step_num, n, 1, 1))
    ###########################################################################
    #                       Pure rolling contact point                        #
    ###########################################################################
    rol_poi_lef_o = np.zeros((plot_step_num, n, 1, 1))
    rol_poi_lef_i = np.zeros((plot_step_num, n, 1, 1))
    rol_poi_rig_o = np.zeros((plot_step_num, n, 1, 1))
    rol_poi_rig_i = np.zeros((plot_step_num, n, 1, 1))
    if Shim_thknss_o > 0 or Shim_thknss_i > 0:
        rol_poi_lef_o_ = np.zeros((plot_step_num, n, 1, 1))
        rol_poi_lef_i_ = np.zeros((plot_step_num, n, 1, 1))
        rol_poi_rig_o_ = np.zeros((plot_step_num, n, 1, 1))
        rol_poi_rig_i_ = np.zeros((plot_step_num, n, 1, 1))
    ###########################################################################
    #                    Relative velocity in contact zone                    #
    ###########################################################################
    if b_r_lub_type == 0:
        rel_vel_b_o = np.zeros((plot_step_num, n, 3, 12))
        rel_vel_b_i = np.zeros((plot_step_num, n, 3, 12))
    else:
        rel_vel_b_o = np.zeros((plot_step_num, n, 3, 720))
        rel_vel_b_i = np.zeros((plot_step_num, n, 3, 720))
    if Shim_thknss_o > 0 or Shim_thknss_i > 0:
        if b_r_lub_type == 0:
            rel_vel_b_o_ = np.zeros((plot_step_num, n, 3, 12))
            rel_vel_b_i_ = np.zeros((plot_step_num, n, 3, 12))
        else:
            rel_vel_b_o_ = np.zeros((plot_step_num, n, 3, 720))
            rel_vel_b_i_ = np.zeros((plot_step_num, n, 3, 720))
    ###########################################################################
    #                             Spin-roll-ratio                             #
    ###########################################################################
    spi_rol_b_o = np.zeros((plot_step_num, n, 1, 1))
    spi_rol_b_i = np.zeros((plot_step_num, n, 1, 1))
    if Shim_thknss_o > 0 or Shim_thknss_i > 0:
        spi_rol_b_o_ = np.zeros((plot_step_num, n, 1, 1))
        spi_rol_b_i_ = np.zeros((plot_step_num, n, 1, 1))
    if b_r_lub_type == 0:
        sli_rol_b_o = np.zeros((plot_step_num, n, 1, 12))
        sli_rol_b_i = np.zeros((plot_step_num, n, 1, 12))
    else:
        sli_rol_b_o = np.zeros((plot_step_num, n, 1, 720))
        sli_rol_b_i = np.zeros((plot_step_num, n, 1, 720))
    if Shim_thknss_o > 0 or Shim_thknss_i > 0:
        sli_rol_b_o_ = np.zeros((plot_step_num, n, 1, 12))
        sli_rol_b_i_ = np.zeros((plot_step_num, n, 1, 12))
    ###########################################################################
    #                              Oil thickness                              #
    ###########################################################################
    film_b_o = np.zeros((plot_step_num, n, 1, 1))
    film_b_i = np.zeros((plot_step_num, n, 1, 1))
    if Shim_thknss_o > 0 or Shim_thknss_i > 0:
        film_b_o_ = np.zeros((plot_step_num, n, 1, 1))
        film_b_i_ = np.zeros((plot_step_num, n, 1, 1))
    ###########################################################################
    #                            Contact position                             #
    ###########################################################################
    re_con_pos_b_o = np.zeros((plot_step_num, n, 3, 1))
    re_con_pos_b_i = np.zeros((plot_step_num, n, 3, 1))
    race_con_pos_b_o = np.zeros((plot_step_num, n, 3, 1))
    race_con_pos_b_i = np.zeros((plot_step_num, n, 3, 1))
    if Shim_thknss_o > 0 or Shim_thknss_i > 0:
        re_con_pos_b_o_ = np.zeros((plot_step_num, n, 3, 1))
        re_con_pos_b_i_ = np.zeros((plot_step_num, n, 3, 1))
        race_con_pos_b_o_ = np.zeros((plot_step_num, n, 3, 1))
        race_con_pos_b_i_ = np.zeros((plot_step_num, n, 3, 1))
    ###########################################################################
    #                             Force-velocity                              #
    ###########################################################################
    if b_r_lub_type == 0:
        trac_coff_b_o = np.zeros((plot_step_num, n, 3, 12))
        trac_coff_b_i = np.zeros((plot_step_num, n, 3, 12))
    else:
        trac_coff_b_o = np.zeros((plot_step_num, n, 3, 720))
        trac_coff_b_i = np.zeros((plot_step_num, n, 3, 720))
    if b_r_lub_type == 0:
        str_slip_b_o = np.zeros((plot_step_num, n, 1, 12))
        str_slip_b_i = np.zeros((plot_step_num, n, 1, 12))
    else:
        str_slip_b_o = np.zeros((plot_step_num, n, 1, 720))
        str_slip_b_i = np.zeros((plot_step_num, n, 1, 720))
    if Shim_thknss_o > 0 or Shim_thknss_i > 0:
        if b_r_lub_type == 0:
            trac_coff_b_o_ = np.zeros((plot_step_num, n, 3, 12))
            trac_coff_b_i_ = np.zeros((plot_step_num, n, 3, 12))
            str_slip_b_o_ = np.zeros((plot_step_num, n, 1, 12))
            str_slip_b_i_ = np.zeros((plot_step_num, n, 1, 12))
        else:
            trac_coff_b_o_ = np.zeros((plot_step_num, n, 3, 720))
            trac_coff_b_i_ = np.zeros((plot_step_num, n, 3, 720))
            str_slip_b_o_ = np.zeros((plot_step_num, n, 1, 720))
            str_slip_b_i_ = np.zeros((plot_step_num, n, 1, 720))
    con_for_slip_b_o = np.zeros((plot_step_num, n, 1, 1))
    con_for_slip_b_i = np.zeros((plot_step_num, n, 1, 1))
    fri_for_slip_b_o = np.zeros((plot_step_num, n, 1, 1))
    fri_for_slip_b_i = np.zeros((plot_step_num, n, 1, 1))
    if Shim_thknss_o > 0 or Shim_thknss_i > 0:
        con_for_slip_b_o_ = np.zeros((plot_step_num, n, 1, 1))
        con_for_slip_b_i_ = np.zeros((plot_step_num, n, 1, 1))
        fri_for_slip_b_o_ = np.zeros((plot_step_num, n, 1, 1))
        fri_for_slip_b_i_ = np.zeros((plot_step_num, n, 1, 1))
    ###########################################################################
    #                             Cage data array                             #
    ###########################################################################
    if n_cseg <= 0:
        #######################################################################
        #                     Ball / ball force-velocity                      #
        #######################################################################
        con_for_slip_b_b = np.zeros((plot_step_num, n, 1, 1))
        fri_for_slip_b_b = np.zeros((plot_step_num, n, 1, 1))
        #######################################################################
        #                Ball / ball contact force and moment                 #
        #######################################################################
        con_for_b_b = np.zeros((plot_step_num, n, 3, 1))
        mom_b_b = np.zeros((plot_step_num, n, 3, 1))
    else:
        #######################################################################
        #                  Ball / cage geometry intereaction                  #
        #######################################################################
        if poc_type <= 3:
            geo_int_b_c = np.zeros((plot_step_num, n, 1, 1))
        else:
            geo_int_b_c = np.zeros((plot_step_num, n, 1, 4))
        #######################################################################
        #                     Ball / cage contact velocity                    #
        #######################################################################
        if poc_type <= 3:
            rel_vel_b_c = np.zeros((plot_step_num, n, 3, 1))
        else:
            rel_vel_b_c = np.zeros((plot_step_num, n, 3, 4))
        #######################################################################
        #                       Ball / cage contact angle                     #
        #######################################################################
        if poc_type <= 3:
            con_ang_b_c = np.zeros((plot_step_num, n, 1, 1))
        else:
            con_ang_b_c = np.zeros((plot_step_num, n, 1, 4))
        #######################################################################
        #                    Ball / cage contact position                     #
        #######################################################################
        if poc_type <= 3:
            re_con_pos_b_c = np.zeros((plot_step_num, n, 3, 1))
            poc_con_pos_b_c = np.zeros((plot_step_num, n, 3, 1))
        else:
            re_con_pos_b_c = np.zeros((plot_step_num, n, 3, 4))
            poc_con_pos_b_c = np.zeros((plot_step_num, n, 3, 4))
        #######################################################################
        #                     Ball / cage force-velocity                      #
        #######################################################################
        if poc_type <= 3:
            con_for_slip_b_p = np.zeros((plot_step_num, n, 1, 1))
            fri_for_slip_b_p = np.zeros((plot_step_num, n, 1, 1))
        else:
            con_for_slip_b_p = np.zeros((plot_step_num, n, 1, 4))
            fri_for_slip_b_p = np.zeros((plot_step_num, n, 1, 4))
        #######################################################################
        #                Ball / cage contact force and moment                 #
        #######################################################################
        if poc_type <= 3:
            con_for_b_c = np.zeros((plot_step_num, n, 3, 1))
            con_for_c_b = np.zeros((plot_step_num, 1, 3, 1))
            mom_b_c = np.zeros((plot_step_num, n, 3, 1))
            mom_c_b = np.zeros((plot_step_num, 1, 3, 1))
        else:
            con_for_b_c = np.zeros((plot_step_num, n, 3, 4))
            con_for_c_b = np.zeros((plot_step_num, 1, 3, 4))
            mom_b_c = np.zeros((plot_step_num, n, 3, 4))
            mom_c_b = np.zeros((plot_step_num, 1, 3, 4))
        #######################################################################
        #                       Cage / race contact angle                     #
        #######################################################################
        con_ang_c_r = np.zeros((plot_step_num, 1, 1, 4))
        #######################################################################
        #                  Cage / race geometry intereaction                  #
        #######################################################################
        geo_int_c_r = np.zeros((plot_step_num, n, 1, 4))
        #######################################################################
        #                        Cage / race velocity                         #
        #######################################################################
        rel_vel_c_r = np.zeros((plot_step_num, 1, 3, 4))
        #######################################################################
        #                    Cage / race contact position                     #
        #######################################################################
        cage_con_pos_c_r = np.zeros((plot_step_num, 1, 3, 4))
        race_con_pos_c_r = np.zeros((plot_step_num, 1, 3, 4))
        #######################################################################
        #                     Cage / race force-velocity                      #
        #######################################################################
        con_for_slip_c_r = np.zeros((plot_step_num, 1, 1, 4))
        fri_for_slip_c_r = np.zeros((plot_step_num, 1, 1, 4))
        #######################################################################
        #                Cage / race contact force and moment                 #
        #######################################################################
        con_for_c_r = np.zeros((plot_step_num, 1, 3, 4))
        con_for_r_c = np.zeros((plot_step_num, 1, 3, 4))
        mom_c_r = np.zeros((plot_step_num, 1, 3, 4))
        mom_r_c = np.zeros((plot_step_num, 1, 3, 4))
    ###########################################################################
    #                           Moving frame force                            #
    ###########################################################################
    if k_brg_mov > 0:
        mov_fr_for_b = np.zeros((plot_step_num, n, 3, 1))
        mov_fr_for_c = np.zeros((plot_step_num, 1, 3, 1))
        mov_fr_for_r = np.zeros((plot_step_num, 1, 3, 2))
    ###########################################################################
    #                          Drag force and moment                          #
    ###########################################################################
    if k_chrn_type > 0:
        chrn_drag_loss_b = np.zeros((plot_step_num, n, 1, 1))
        chrn_drag_loss_c = np.zeros((plot_step_num, n, 1, 1))
        for_drag_b = np.zeros((plot_step_num, n, 1, 1))
        mom_drag_b = np.zeros((plot_step_num, n, 3, 1))
        mom_drag_c = np.zeros((plot_step_num, 1, 3, 3))
    ###########################################################################
    #                          Drag force and moment                          #
    ###########################################################################
    if k_life_freq > 0:
        #######################################################################
        #                        Bearing fatigued life                        #
        #######################################################################
        fati_life_brg = np.zeros((plot_step_num, 1, 1, 3))
        fati_life_brg_LP = np.zeros((plot_step_num, 1, 1, 3))
        fati_life_brg_GZ = np.zeros((plot_step_num, 1, 1, 3))
        fati_life_brg_IH = np.zeros((plot_step_num, 1, 1, 3))
        #######################################################################
        #                         Race fatigued life                          #
        #######################################################################
        fati_life_r = np.zeros((plot_step_num, 2, 1, 3))
        fati_life_r_LP = np.zeros((plot_step_num, 2, 1, 3))
        fati_life_r_GZ = np.zeros((plot_step_num, 2, 1, 3))
        fati_life_r_IH = np.zeros((plot_step_num, 2, 1, 3))
        #######################################################################
        #                         Race fatigued life                          #
        #######################################################################
        fati_life_b_LP = np.zeros((plot_step_num, n, 1, 3))
        fati_life_b_GZ = np.zeros((plot_step_num, n, 1, 3))
        fati_life_b_IH = np.zeros((plot_step_num, n, 1, 3))
###############################################################################
#                                 Delete data                                 #
###############################################################################
for i in range(len(base_data[0])):
    list_key = list(base_data[i+1].keys())
    for j in range(len(list_key)):
        del globals()[list_key[j]]
    del globals()[base_data[0][i]]
for i in range(len(mod_index_name)):
    del globals()[mod_index_name[i]]
del (F_max, f, i, j, element, elements, data, file,
     line, row, rows, row_data, brg_type, list_key,
     plot_step_num, base_data, base_data_name,
     mod_index_data, mod_name, mod_index_name)
# Delete almost global variables