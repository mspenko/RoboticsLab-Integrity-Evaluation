import numpy as np
from scipy.stats import norm


class IntegrityDataClass:
      time = None # time at which each data has been stored
      association = None # associations without zeros
      association_full = None # associations at k w/o LM selection
      P_MA_k = None # probability of missassociations at k
      P_MA_k_full = None # probability of missassociations at k w/o LM selection
      P_H = None # hypotheses probabilities at k
      detector = None
      detector_threshold = None
      p_hmi = None
      n_L_M = None
      M = None
      sigma_hat = None 
      p_eps = None # prob that the estimate is out of bounds w/out faults

      def __init__(self,n):
          # allocate memory for approximately the number of readings
          self.association= [None]*n; 
          self.association_full= [None]*n; 
          self.P_MA_k= [None]*n;
          self.P_MA_k_full= [None]*n; 
          self.P_H= [None]*n; 
          self.detector= np.zeros((n,1));
          self.detector_threshold= np.zeros((n,1));
          self.p_hmi= np.zeros((n,1));
          self.n_L_M= np.zeros((n,1));
          self.sigma_hat= np.zeros((n,1));
          self.time= np.zeros((n,1));
          self.p_eps= np.zeros((n,1));
          self.M= np.zeros((n,1));

      def store(self, im, estimator, counters, params):
          self.p_hmi[counters.k_im]= im.p_hmi;
          self.n_L_M[counters.k_im]= im.n_L_M;
          self.P_H[counters.k_im]= im.P_H;
          self.detector_threshold[counters.k_im]= im.T_d**2;
          self.sigma_hat[counters.k_im]= im.sigma_hat;
          self.time[counters.k_im]= counters.time_sim;
          self.p_eps[counters.k_im]= 2* normcdf(-params.alert_limit, 0, im.sigma_hat);
          self.M[counters.k_im]= im.M;
            
          if (params.SWITCH_FACTOR_GRAPHS == 0):      
             self.association[counters.k_im]= estimator.association_no_zeros;
             self.association_full[counters.k_im]= estimator.association_full;
             self.P_MA_k[counters.k_im]= im.P_MA_k;
             self.P_MA_k_full[counters.k_im]= im.P_MA_k_full;    
             self.detector[counters.k_im]= im.q_M;
            
        
       
      def delete_extra_allocated_memory(self, counters):
          self.association =np.delete(self.association,np.s_[counters.k_im:])
          self.association_full = np.delete(self.association_full,np.s_[counters.k_im:])
          self.P_MA_k = np.delete(self.P_MA_k,np.s_[counters.k_im:])
          self.P_MA_k_full = np.delete(self.P_MA_k_full,np.s_[counters.k_im:])         
          self.P_H = np.delete(self.P_H,np.s_[counters.k_im:])
          self.detector = np.delete(self.detector,np.s_[counters.k_im:])
          self.detector_threshold = np.delete(self.detector_threshold,np.s_[counters.k_im:])
          self.p_hmi = np.delete(self.p_hmi,np.s_[counters.k_im:])
          self.n_L_M = np.delete(self.n_L_M,np.s_[counters.k_im:])
          self.sigma_hat = np.delete(self.sigma_hat,np.s_[counters.k_im:])
          self.time = np.delete(self.time,np.s_[counters.k_im:])
          self.p_eps = np.delete(self.p_eps,np.s_[counters.k_im:])
          self.M = np.delete(self.M,np.s_[counters.k_im:])
        
    
