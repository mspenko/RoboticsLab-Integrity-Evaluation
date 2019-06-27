import math
import numpy as np
import sys
sys.path.insert(0,'../functions/')
import ecef2lla
from scipy.linalg import block_diag

class GPSClass:
    
      num_readings = None
      time = None
      timeInit = None
      R = None
      msmt = None
      R_NE = None
      IS_GPS_AVAILABLE = None 

      def __init__(self,timeStatic,params):

          self.IS_GPS_AVAILABLE = 0
          if(params.SWITCH_SIM == 1):
             self.time = 0
             return 0
          numEpochStaticGPS = math.ceil(timeStatic)
          dtype1 = np.dtype([('0','f8'),('1','f8'),('2','f8'),('3','f8'),('4','f8'),('5','f8'),('6','f8'),('7','f8'),('8','f8'),('9','f8'),('10','f8')])
          data = np.loadtxt(params.file_name_gps)
          self.num_readings= data.shape[0]
          self.time = data[:,3]
          posX=     data[:,4]
          posY=     data[:,5]
          posZ=     data[:,6]
          velX=     data[:,7]
          velY=     data[:,8]
          velZ=     data[:,9]
          sigPosX=  data[:,10]
          sigPosY=  data[:,11]
          sigPosZ=  data[:,12]
          sigVelX=  data[:,13]
          sigVelY=  data[:,14]
          sigVelZ=  data[:,15]

          # Save the initial time as reference for other sensors
          self.timeInit= self.time[1]
            
          # Make time start at zero
          self.time= self.time - self.time[0]
            
          # create variables
          self.msmt= np.transpose(np.array([posX, posY, posZ, velX, velY, velZ]))
          self.R= np.transpose((np.array([sigPosX, sigPosY, sigPosZ, sigVelX, sigVelY, sigVelZ])**2))
          
          # Use initial position as reference
          muX= np.mean(posX[0:numEpochStaticGPS])
          muY= np.mean(posY[0:numEpochStaticGPS])
          muZ= np.mean(posZ[0:numEpochStaticGPS])
          self.msmt[:,0]= self.msmt[:,0] - muX
          self.msmt[:,1]= self.msmt[:,1] - muY
          self.msmt[:,2]= self.msmt[:,2] - muZ
          # Convert from ECEF to Navigation-frame
          tmp =  ecef2lla.ecef2lla(muX,muY,muZ);
          phi,lambdal = tmp[0],tmp[1]
            
          self.R_NE= np.array([[-math.sin(phi)*math.cos(lambdal),-math.sin(phi)*math.sin(lambdal),math.cos(phi)],
                      [-math.sin(lambdal), math.cos(lambdal), 0],
                      [-math.cos(phi)*math.cos(lambdal), -math.cos(phi)*math.sin(lambdal), -math.sin(phi)]])
          
          R_NE_block= block_diag( self.R_NE, self.R_NE );
          self.msmt= np.dot(R_NE_block,np.transpose(self.msmt))
            
          for i in range((self.time).shape[0]):
              self.R[i,:]= np.diag( R_NE_block * np.diag( self.R[i,:] ) * np.transpose(R_NE_block) )
          # increase GPS variance
          self.R[:,0:3]= self.R[:,0:3]*(params.mult_factor_pose_gps**2); ################## CAREFUL
          self.R[:,3:6]= self.R[:,3:6]*(params.mult_factor_vel_gps**2);  ################## CAREFUL
          
     
