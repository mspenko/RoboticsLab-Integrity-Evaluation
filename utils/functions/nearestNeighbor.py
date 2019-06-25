import numpy as np
import math

def nearestNeighbor(z,appearances,R,T,T_newLM):
    global XX,PX

    n_F= z.shape[1]
    n_L= (XX.shape[0] - 15) / 2;
    association= np.ones(1,n_F) * (-1)

    if (n_F == 0 or n_L == 0):
        return 0

    spsi= math.sin(XX[8])
    cpsi= math.cos(XX[8])
    zHat= np.zeros((2,1))
    # Loop over extracted features
    for i in range(n_F):
        minY= T_newLM
        for l in range(n_L):
            ind= np.arange(14 + (2*l-1)-1,(15 + 2*l)-1,dtype=np.int);
            dx= XX[ind[0] - XX[0]];
            dy= XX[ind[1] - XX[1]];
        
            zHat[0]=  dx*cpsi + dy*spsi;
            zHat[1]= -dx*spsi + dy*cpsi
            gamma= np.transpose(z[i,:]) - zHat
         
            H= np.array([[-cpsi, -spsi, -dx*spsi + dy*cpsi,  cpsi, spsi],[spsi, -cpsi, -dx*cpsi - dy*spsi, -spsi, cpsi]])
            Y= H * PX[[0,1,8,ind],[0,1,8,ind]] * np.transpose(H) + R
            y2= np.transpose(gamma) / Y * gamma
            if (y2 < minY):
               minY= y2
               association[i]= l
    
        # If the minimum value is very large --> new landmark
        if (minY > T & minY < T_newLM):
           association[i]= 0


    # Increase appearances counter  
    for i in range(n_F):
        if (association[i] != -1 & association[i] != 0):
           appearances[association[i]]= appearances[association[i]] + 1

    return [association,appearances]

