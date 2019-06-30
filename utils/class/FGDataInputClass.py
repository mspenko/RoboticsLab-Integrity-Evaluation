import math
import numpy as np
import sys

class FGDataInputClass:
      gps_msmt = None
      gps_R = None
      lidar = None
      imu = None
      pose = None
      associations = None

      def __init__(self,num_readings):
 
          # allocate memory
          self.lidar= [None]*num_readings;
          self.imu= [None]*num_readings;
          self.gps_msmt= [None]*num_readings;
          self.gps_R= [None]*num_readings;
          self.pose= [None]*num_readings;
          self.associations= [None]*num_readings;

      def delete_fields_corresponding_to_static_epochs(self, lidar):
          self.lidar = np.delete(self.lidar,np.s_[1:lidar.index_of_last_static_lidar_epoch],axis =0)
          self.imu = np.delete(self.imu,np.s_[1:lidar.index_of_last_static_lidar_epoch],axis =0)
          self.gps_msmt = np.delete(self.gps_msmt,np.s_[1:lidar.index_of_last_static_lidar_epoch],axis =0)
          self.gps_R = np.delete(self.gps_R,np.s_[1:lidar.index_of_last_static_lidar_epoch],axis =0)
          self.pose = np.delete(self.pose,np.s_[1:lidar.index_of_last_static_lidar_epoch],axis =0)
          self.associations = np.delete(self.associations,np.s_[1:lidar.index_of_last_static_lidar_epoch],axis =0)
