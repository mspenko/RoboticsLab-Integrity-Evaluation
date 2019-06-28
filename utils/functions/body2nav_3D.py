import numpy as np
import R_NB_rot

def body2nav_3D(z,x):
    # Convert in 3D
    R_NB= R_NB_rot.R_NB_rot(x[6],x[7],x[8]);
    z= np.transpose( np.dot(R_NB,np.transpose(z)) + x[0:3] );

    # Put the back into 2D
    z= z[:,1:2];

    return z
