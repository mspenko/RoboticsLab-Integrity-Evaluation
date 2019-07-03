

class CountersClass:

      time_sim= 0
      time_sum= 0
      time_sum_virt_z= 0
      time_sum_virt_y= 0
      time_gps = None
      time_lidar = None
        
      k_update = 0
      k_gps = 0
      k_lidar = 0
      k_im = 0

      def __init__(self,gps,lidar,params):
          # if it's a simulation --> time is not read from files
          if (params.SWITCH_SIM == 0 & params.SWITCH_FACTOR_GRAPHS == 0):
              self.time_gps= gps.time[0] # this is zero, as the GPS time is the reference
              self.time_lidar= lidar.time[0,1]
            
        
        
      def increase_gps_counter(self):
          self.k_gps= self.k_gps + 1
        
        
      def increase_lidar_counter(self):
          self.k_lidar= self.k_lidar + 1
        
        
      def increase_integrity_monitoring_counter(self):
          self.k_im= self.k_im + 1
        
        
      def increase_time_sums(self, params):
          self.time_sum= self.time_sum + params.dt_imu
          self.time_sum_virt_z= self.time_sum_virt_z + params.dt_imu
          self.time_sum_virt_y= self.time_sum_virt_y + params.dt_imu           
        
        
      def increase_time_sum_sim(self, params):
          self.time_sum= self.time_sum + params.dt_sim
        
        
      def increase_time_sim(self, params):
          self.time_sim= self.time_sim + params.dt_sim
        
        
      def reset_time_sum(self):
          self.time_sum= 0
        
        
      def reset_time_sum_virt_z(self):
          self.time_sum_virt_z= 0
        
        
      def reset_time_sum_virt_y(self):
          self.time_sum_virt_y= 0
        
          
