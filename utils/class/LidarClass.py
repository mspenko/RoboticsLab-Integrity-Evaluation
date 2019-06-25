import yaml
import numpy as np
from scipy.linalg import block_diag
from scipy.linalg import sqrtm
from scipy.linalg import inv
from scipy.stats.distributions import chi2
import math
import sys
sys.path.insert(0,'../functions/')
import body2nav_3D



class LidarClass():
      
      time = np.array([])
      num_readings = None
      R = None
      areas_to_remove = None
      msts = None # this only stores the current msmt -->  private
      index_of_last_static_lidar_epoch = None # index of last static lidar epoch
    

      def __init__(self,params,init_time):
          # Initialization (it will be obatined during the run)
          self.index_of_last_static_lidar_epoch= 0
            
          if (params.SWITCH_SIM==1):
             self.time= np.array()
             return 0

          # substracts the GPS start time which is used as reference    
          # load the variable "T_LIDAR"
          dtype1 = np.dtype([('epoch','f8'),('time','f8')])
          self.time = np.loadtxt(params.file_name_lidar_path+'T_LIDAR.txt',dtype = dtype1)
          try:
            dtype2 = np.dtype([('xmin','I8'),('xmax','I8'),('ymin','I8'),('ymax','I8')])
            self.areas_to_remove = np.loadtxt(params.file_name_lidar_path+'T_LIDAR.txt',dtype = dtype2)
          except:
             print("[MSG] no area to remove")

          #Use the GPS first reading time as reference
          self.time[:][1]= self.time[:][1] - init_time
            
          #If some of the initial times are negative (prior to first GPS reading) --> eliminate them
          self.time[self.time[:,1]<0] [ :]= [] 
          #number of lidar scans
          self.num_readings= length(self.time)


      # ----------------------------------------------
      # ----------------------------------------------
      def get_msmt(self,epoch,params):
            # load the mat file with the extrated features at the lidar epoch specified
            dtype2 = np.dtype([('x','f8'),('y','f8')])
            fileName = params.file_name_lidar_path+'matFiles/Epoch'+str(epoch)+'.txt'
            # loads the z variable with range and bearing
            self.msmt= np.loadtxt(fileName,dtype3) 
            # if there are features --> prepare the measurement
            if (self.msmt != None):
                if (params.SWITCH_REMOVE_FAR_FEATURES==1):
                    self.remove_far_features(params.lidarRange)
                # Add height
                self.msmt= np.array([self.msmt, np.ones((size(self.msmt,1),1)) * params.feature_height]) 
            
      # ----------------------------------------------
      # ----------------------------------------------
      def remove_far_features(self,lidarRange):
          # removes extracted features farther than "lidarRange"
          d= self.msmt[:][0]**2 + self.msmt[:][1]**2 
          self.msmt[ d > lidarRange**2][:]= [] 
      # ----------------------------------------------
      # ----------------------------------------------
      def remove_features_in_areas(self,x):
            # remove features from areas, each area is: [minX, maxX, minY, maxY]
            
            # Remove people-features for the data set 
            counter = 0
            while(i< self.areas_to_remove.shape[0]):
                area= self.areas_to_remove[i,:] 
                
                # transform to nav-frame first
                msmt_nav_frame= body2nav_3D(self.msmt,x)
                
                # Remove people-features
                inX  = []
                for i in msmt_nav_frame:
                    if(msmt_nav_frame[i,0]>area[0]):
                       if(msmt_nav_frame[i,0] < area[1]): #inX= (msmt_nav_frame(:,0) > area(0)) & (msmt_nav_frame(:,0) < area(1)) 
                          inX.append(1)
                       else:
                          inX.append(0)
                inY = []
                for i in msmt_nav_frame:
                    if(msmt_nav_frame[i,1]>area[2]):
                       if(msmt_nav_frame[i,1] < area[3]): #inY= (msmt_nav_frame(:,1) > area(2)) & (msmt_nav_frame(:,1) < area(3)) 
                          inY.append(1)
                       else:
                          inY.append(0)
                inX,inY = np.array(inX),np.array(inY)

                tmp_list = []
                for i in inX:
                    if (inX[i] == inY[i]):          #self.msmt( inX & inY, :)= [] 
                       tmp_list.append(1)
                      
                self.msmt = np.array(tmp_list)
                counter = counter+1 
