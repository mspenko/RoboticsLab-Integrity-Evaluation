import yaml
import numpy as np
from scipy.linalg import block_diag
from scipy.linalg import sqrtm
from scipy.linalg import inv
from scipy.stats.distributions import chi2
import scipy.io as sio
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
          #self.time = np.loadtxt(params.file_name_lidar_path+'T_LIDAR.mat')
          self.time = sio.loadmat(params.file_name_lidar_path+'T_LIDAR.mat')
          self.time = self.time['T_LIDAR']
          try:
            #self.areas_to_remove = np.loadtxt(params.file_name_lidar_path+'areas_to_remove.mat')
             self.areas_to_remove = sio.loadmat(params.file_name_lidar_path+'areas_to_remove.mat')
             self.areas_to_remove = self.areas_to_remove['areas_to_remove']
          except:
             print("[MSG] no area to remove")
          #Use the GPS first reading time as reference
          self.time[:,1]= self.time[:,1] - np.ones(self.time.shape[0])*init_time
          
          #If some of the initial times are negative (prior to first GPS reading) --> eliminate them
          acc = 0
          for i in self.time[:,1]:
              if(i<0):
                 np.delete(self.time,acc,axis = 0)
              acc = acc+1
          #number of lidar scans
          self.num_readings= self.time.shape[0]


      # ----------------------------------------------
      # ----------------------------------------------
      def get_msmt(self,epoch,params):

            # load the mat file with the extrated features at the lidar epoch specified
            fileName = params.file_name_lidar_path+'textFiles/Epoch'+str(int(epoch))+'.txt'
            # loads the z variable with range and bearing
            self.msmt= np.loadtxt(fileName)
            if not (type(self.msmt[0]) is np.ndarray ):
               self.msmt= np.array([self.msmt])
            # if there are features --> prepare the measurement
            if (self.msmt.shape[0] !=0):
                if (params.SWITCH_REMOVE_FAR_FEATURES==1):
                    self.remove_far_features(params.lidarRange)
                    if (len(self.msmt)==0):
                      return 0
                # Add height
                #self.msmt= np.array([self.msmt, np.ones((self.msmt.shape[1],1)) * params.feature_height])
                self.msmt = np.concatenate((self.msmt,np.dot(np.ones((self.msmt.shape[0],1)), params.feature_height)),axis = 1)
      # ----------------------------------------------
      # ----------------------------------------------
      def remove_far_features(self,lidarRange):
          # removes extracted features farther than "lidarRange"
          if (self.msmt).shape[0] == 1:
            d= self.msmt[0,0]**2 + self.msmt[0,1]**2 
          else:
            d= self.msmt[:][0]**2 + self.msmt[:][1]**2 
          
          check= np.shape(d)
          notScalar= len(check)
          if (notScalar == 0):
            if(d>lidarRange**2):
              self.msmt=[]
              return 0
          else:
            tmp_list = []
            for i in range(len(d)):
                if(d[i]>lidarRange**2):
                  tmp_list.append(i)
            self.msmt =np.delete(self.msmt,tmp_list,axis = 0)
          #self.msmt[ d > lidarRange**2][:]= [] 
      # ----------------------------------------------
      # ----------------------------------------------
      def remove_features_in_areas(self,x):
            # remove features from areas, each area is: [minX, maxX, minY, maxY]
            # Remove people-features for the data set 
            i = 0
            if (len(self.msmt) == 0):
              return 0
            while( i< self.areas_to_remove.shape[0]):
                area= self.areas_to_remove[i,:] 
                # transform to nav-frame first
                msmt_nav_frame= body2nav_3D.body2nav_3D(self.msmt,x)
                
                # Remove people-features
                #inX= (msmt_nav_frame(:,0) > area(0)) & (msmt_nav_frame(:,0) < area(1)) 
                inX = []
                for j in msmt_nav_frame[:,0]:
                    if(j>area[0] and j<area[1]):
                      inX.append(1)
                    else:
                      inX.append(0)

                #inY= (msmt_nav_frame(:,1) > area(2)) & (msmt_nav_frame(:,1) < area(3))
                inY = []
                for j in msmt_nav_frame[:,1]:
                    if(j>area[2] and j<area[3]):
                      inY.append(1)
                    else:
                      inY.append(0)

                inX,inY = np.array(inX),np.array(inY)

                tmp_list = []
                for j in range(len(inX)):
                    if (inX[j] ==1 and inX[j] == inY[j]):          #self.msmt( inX & inY, :)= [] 
                       tmp_list.append(j)
                self.msmt = np.delete(self.msmt,tmp_list,axis = 0)      
                i = i+1 
