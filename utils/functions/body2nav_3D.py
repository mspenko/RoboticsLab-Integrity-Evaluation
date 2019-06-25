import numpy as np
import R_NB_rot

def body2nav_3D(z,x):
    # Convert in 3D
    R_NB= R_NB_rot.R_NB_rot(x[7-1],x[8-1],x[9-1]);
    z= np.transpose( R_NB * np.transpose(z) + x[1:3] );

    # Put the back into 2D
    z= z[:,1:2];

    return z
