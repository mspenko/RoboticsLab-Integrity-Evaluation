import numpy as np
import math

def ecef2lla(x,y,z):

    # WGS84 ellipsoid constants:
    a = 6378137
    e = 8.1819190842622e-2

    # calculations:
    b   = sqrt(a**2*(1-e**2))
    ep  = sqrt((a**2-b**2)/b**2)
    p   = sqrt(x**2+y**2)
    th  = math.atan2(a*z,b*p)
    lon = math.atan2(y,x)
    lat = math.atan2((z+ep**2*b*math.sin(th)**3),(p-e**2*a*math.cos(th)**3))
    N   = a/sqrt(1-e**2*math.sin(lat)**2)
    alt = p/math.cos(lat)-N

    # return lon in range [0,2*pi)
    lon = np.mod(lon,2*math.pi)

    # correct for numerical instability in altitude near exact poles:
    # (after this correction, error is about 2 millimeters, which is about
    # the same as the numerical precision of the overall function)
 
    k= abs(x)<1 & abs(y)<1
    alt[k] = abs(z[k])-b

    return [lat,lon,alt]
