import cmath
from math import ceil
from numpy import rad2deg, deg2rad, angle
import numpy as np

counterlockwise = lambda x: x if x > 0 else 360 - abs(x)
def relativeAngle(x, y):
    rel = (abs(counterlockwise(x) - counterlockwise(y)))
    rel = rel if rel < 180 else 360 - rel
    return rel

# 0 if it is 0 sequence
# 1 if positive sequence 
# -1 if negative sequence 
def threephaseSeq(a, b, c):
    rel_a_b = abs(counterlockwise(a) - counterlockwise(b))
    rel_a_c = abs(counterlockwise(a) - counterlockwise(c))
    if rel_a_c > rel_a_b: return 1
    if rel_a_c < rel_a_b: return -1
    return 0

def getPhaseAngleBySite(gen_signal, phased_angles_mapper):
    angle_signal = gen_signal.replace('Voltage Mag', 'Voltage Ang')    
    return phased_angles_mapper[angle_signal]

def pos_seq(mag1, ang1, mag2, ang2, mag3, ang3):
    e1 = cmath.rect(mag1, np.deg2rad(ang1))
    e2 = cmath.rect(mag2, np.deg2rad(ang2))
    e3 = cmath.rect(mag3, np.deg2rad(ang3))
    a = cmath.rect(1.0, np.deg2rad(120))

    V1 = (e1 + a*e2 + a**2*e3)
    return round(abs(V1)/3,2), round(np.rad2deg(np.angle(V1)))

def neg_seq(mag1, ang1, mag2, ang2, mag3, ang3):
    e1 = cmath.rect(mag1, np.deg2rad(ang1))
    e2 = cmath.rect(mag2, np.deg2rad(ang2))
    e3 = cmath.rect(mag3, np.deg2rad(ang3))
    a = cmath.rect(1.0, np.deg2rad(120))
    
    V2 = (e1 + a**2*e2 + a*e3)
    return round(abs(V2)/3,2), round(np.rad2deg(np.angle(V2)))

def zero_seq(mag1, ang1, mag2, ang2, mag3, ang3):
    e1 = cmath.rect(mag1, np.deg2rad(ang1))
    e2 = cmath.rect(mag2, np.deg2rad(ang2))
    e3 = cmath.rect(mag3, np.deg2rad(ang3))
    a = cmath.rect(1.0, np.deg2rad(120))
    
    V0 = (e1 + e2 + e3)
    return round(abs(V0)/3,2), round(np.rad2deg(np.angle(V0)))

def a_seq(mag1, ang1, mag2, ang2, mag3, ang3):
    v0, a0 = zero_seq(mag1, ang1, mag2, ang2, mag3, ang3)
    v1, a1 = pos_seq(mag1, ang1, mag2, ang2, mag3, ang3)
    v2, a2 = neg_seq(mag1, ang1, mag2, ang2, mag3, ang3)
    
    e0 = cmath.rect(v0, np.deg2rad(a0))
    e1 = cmath.rect(v1, np.deg2rad(a1))
    e2 = cmath.rect(v2, np.deg2rad(a2))
    a = cmath.rect(1.0, np.deg2rad(120))
    
    V0 = (e0 + e1 + e2)
    return round(abs(V0),2), round(np.rad2deg(np.angle(V0)))

def b_seq(mag1, ang1, mag2, ang2, mag3, ang3):
    v0, a0 = zero_seq(mag1, ang1, mag2, ang2, mag3, ang3)
    v1, a1 = pos_seq(mag1, ang1, mag2, ang2, mag3, ang3)
    v2, a2 = neg_seq(mag1, ang1, mag2, ang2, mag3, ang3)
    
    e0 = cmath.rect(v0, np.deg2rad(a0))
    e1 = cmath.rect(v1, np.deg2rad(a1))
    e2 = cmath.rect(v2, np.deg2rad(a2))
    a = cmath.rect(1.0, np.deg2rad(120))
    
    V1 = (e0 + a**2*e1 + a*e2)
    return round(abs(V1),2), round(np.rad2deg(np.angle(V1)))

def c_seq(mag1, ang1, mag2, ang2, mag3, ang3):
    v0, a0 = zero_seq(mag1, ang1, mag2, ang2, mag3, ang3)
    v1, a1 = pos_seq(mag1, ang1, mag2, ang2, mag3, ang3)
    v2, a2 = neg_seq(mag1, ang1, mag2, ang2, mag3, ang3)
    
    e0 = cmath.rect(v0, np.deg2rad(a0))
    e1 = cmath.rect(v1, np.deg2rad(a1))
    e2 = cmath.rect(v2, np.deg2rad(a2))
    a = cmath.rect(1.0, np.deg2rad(120))
    
    V2 = (e0 + a*e1 + a**2*e2)
    return round(abs(V2),2), round(np.rad2deg(np.angle(V2)))