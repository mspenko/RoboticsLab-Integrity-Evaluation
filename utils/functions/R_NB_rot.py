import numpy as np
import math

def R_NB_rot(phi,theta,psi):
    
    R_BN = np.array([[math.cos(psi)*math.cos(theta), math.sin(psi)*math.cos(theta), -math.sin(theta)],
                      [-math.sin(psi)*math.cos(phi) + math.cos(psi)*math.sin(theta)*math.sin(phi), math.cos(psi)*math.cos(phi) + math.sin(psi)*math.sin(theta)*math.sin(phi), math.cos(theta)*math.sin(phi)],
                      [math.sin(psi)*math.sin(phi) + math.cos(psi)*math.sin(theta)*math.cos(phi), -math.cos(psi)*math.sin(phi) + math.sin(psi)*math.sin(theta)*math.cos(phi), math.cos(theta)*math.cos(phi)]])
    
    R_NB = np.transpose(R_BN)

    return R_NB
