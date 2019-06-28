import numpy as np

class UpdateDataClass:

      x_true = None
      error = None
      error_state_interest = None
      sig_state_interest = None
      XX = None
      PX = None
      time = None
      miss_associations = None
      num_associated_lms = None
      num_of_extracted_features = None
      q_d = None # detectors
      T_d = None # detector thresholds
      n_L_k = None # number of associated features at k
      n_L_M = None # number of associated features in the ph
      num_faults = None # number of injected faults 
      odometry = None # [velocity, steering angle]

      def __init__(self,num_readings, params):
          # allocate memory
          self.x_true= np.zeros((params.m, num_readings))
          self.error= np.zeros((params.m, num_readings))
          self.error_state_interest= np.zeros((num_readings, 1))
          self.sig_state_interest= np.zeros((num_readings, 1))
          self.XX= np.zeros((params.m, num_readings))
          self.PX= np.zeros((params.m, num_readings))
          self.time= np.zeros((num_readings, 1))
          self.miss_associations= np.zeros((num_readings, 1))
          self.num_associated_lms= np.zeros((num_readings, 1))
          self.num_of_extracted_features= np.zeros((num_readings, 1))
          self.q_d= np.zeros((num_readings, 1))
          self.T_d= np.zeros((num_readings, 1) )
          self.n_L_M= np.zeros((num_readings, 1))
          self.n_L_k= np.zeros((num_readings, 1))
          self.num_faults= np.zeros((num_readings, 1))
          self.odometry= np.zeros((2, num_readings))
      
      # ----------------------------------------------
      # ----------------------------------------------

      def store(self, epoch, estimator, time):
          self.XX[:,epoch]= (estimator.XX[0:15]).transpose()
          self.PX[:,epoch]= np.diag( estimator.PX[0:15,0:15] ) # store only variances
          self.time[epoch]= time
          self.num_associated_lms[epoch]= estimator.num_associated_lms
      # ----------------------------------------------
      # ----------------------------------------------
      def store_sim(self, epoch, estimator, time, params):
            self.error[:,epoch]= estimator.XX - estimator.x_true
            self.XX[:,epoch]= estimator.XX
            self.PX[:,epoch]= np.diag( estimator.PX ) # store only variances
            self.time[epoch]= time
            self.miss_associations[epoch]= sum((estimator.association != estimator.association_true * estimator.association) )
            self.num_associated_lms[epoch]= estimator.num_associated_lms
            self.num_of_extracted_features[epoch]= estimator.num_of_extracted_features
      # ----------------------------------------------
      # ----------------------------------------------
      def store_fg(self, epoch, estimator, time, params):
            estimator.compute_alpha(params)
            
            self.x_true[:,epoch]= estimator.x_true
            self.XX[:,epoch]= estimator.XX
            self.error[:,epoch]= estimator.XX - estimator.x_true
            self.error_state_interest[epoch]= np.transpose(estimator.alpha)* (estimator.XX - estimator.x_true)
            self.sig_state_interest[epoch]= np.sqrt( np.transpose(estimator.alpha)* estimator.PX * estimator.alpha )
            self.PX[:, epoch]= np.diag( estimator.PX )
            self.time[epoch]= time
            self.num_associated_lms[epoch]= estimator.n_L_k
            self.q_d[epoch]= estimator.q_d
            self.T_d[epoch]= estimator.T_d
            self.n_L_k[epoch]= estimator.n_L_k
            self.n_L_M[epoch]= estimator.n_L_M
            if (params.SWITCH_OFFLINE == 0):
                self.num_faults[epoch]= estimator.num_faults_k
                self.odometry[:, epoch]= estimator.odometry_k
      # ----------------------------------------------
      # ----------------------------------------------
      def delete_extra_allocated_memory(self, counters):
          self.x_true = np.delete(self.true,np.s_[:, counters.k_update:],1)
          self.XX = np.delete(self.XX,np.s_[:, counters.k_update:],1)                
          self.error = np.delete(self.error,np.s_[:, counters.k_update:],1)                
          self.error_state_interest = np.delete(self.error_state_interest,np.s_[counters.k_update:])                
          self.sig_state_interest = np.delete(self.sig_state_interest,np.s_[counters.k_update:])                
          self.PX = np.delete(self.PX,np.s_[:, counters.k_update:],1)                
          self.time = np.delete(self.time,np.s_[counters.k_update:])                
          self.num_associated_lms = np.delete(self.num_associated_lms,np.s_[counters.k_update:])                
          self.q_d = np.delete(self.q_d,np.s_[counters.k_update:])                
          self.T_d = np.delete(self.T_d,np.s_[counters.k_update:])                
          self.n_L_k = np.delete(self.n_L_k,np.s_[counters.k_update:])                
          self.n_L_M = np.delete(self.n_L_M,np.s_[counters.k_update:])                
          self.num_faults = np.delete(self.num_faults,np.s_[counters.k_update:])                
          self.odometry = np.delete(self.odometry,np.s_[:, counters.k_update:],1)               

         
