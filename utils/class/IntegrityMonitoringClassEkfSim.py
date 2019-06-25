import math
import numpy as np


class IntegrityMonitoringClassEkfSim:
      m = 3
      calculate_A_M_recursively = 0
      C_req = None
      p_hmi = None
      detector_threshold = None  
      is_extra_epoch_needed= -1 # initialize as (-1), then a boolean
      ind_im= [0,1,2]
      # (maybe unnecessary)
      E = None
      B_bar = None
     
      # hypotheses
      inds_H = None # faulted indexes under H hypotheses
      P_H_0 = None
      P_H = None
      T_d = None
      n_H = None
      n_max = None
        
      # for MA purposes
      mu_k = None
      kappa = None

      # current-time (k) only used when needed to extract elements
      sigma_hat = None
      Phi_k = None
      H_k = None
      L_k = None
      Lpp_k = None
      P_MA_k = None
      P_MA_k_full = None
        
      # augmented (M) 
      M= 0  # size of the preceding horizon in epochs
      n_M = None   # num msmts in the preceding horizon (including k) -if FG ---> num abs msmts
      n_L_M = None # num landmarks in the preceding horizon (including k)
      Phi_M = None
      q_M = None
      gamma_M = None
      Y_M = None
      A_M = None
      M_M = None
      P_MA_M = None
      P_F_M = None
        
      # preceding horizon saved (ph)
      Phi_ph = None
      q_ph = None
      gamma_ph = None
      A_ph = None
      L_ph = None
      Lpp_ph = None
      H_ph = None
      Y_ph = None
      P_MA_ph = None
      n_ph = None
      n_F_ph = None # number of features associated in the preceding horizon
        
      # Factor Graph variables
      m_M = None       # number of states to estimate
      n_total = None   # total number of msmts (prior + relative + abs)
      XX_ph = None
      D_bar_ph = None
      A = None
      Gamma_fg = None # information matrix
      M_fg = None
      PX_prior = None
      PX_M = None
      abs_msmt_ind = None
      faulted_LMs_indices = None
      Gamma_prior = None
      lidar_msmt_ind = None
      gps_msmt_ind = None
      n_gps_ph = None # number of gps msmt at each epoch in PH
      H_gps_ph = None
      H_lidar_ph = None
      n_M_gps = None
      A_reduced = None
      min_f_dir_vs_M_dir = None
      f_mag = None
        
      noncentral_dof= [None]*10000
      f_dir_sig2= [None]*10000
      M_dir= [None]*10000
      counter_H=0
      #=============================================================================================================================================
      def __init__(self,params, estimator):
          # if the preceding horizon is fixed in epochs --> set M
          if (params.SWITCH_FIXED_LM_SIZE_PH == 1):
             self.M= 0
          else:
             self.M= params.M
          # continuity requirement
          self.C_req= params.continuity_requirement
          # initialize the preceding horizon
          self.n_ph = np.zeros( (params.M, 1) );
          self.Phi_ph = [None]*(params.M + 1)   #cell( 1, params.M + 1 ); # need an extra epoch here
          self.H_ph =  [None]*params.M  #cell( 1, params.M );
          self.gamma_ph = [None]*params.M # cell(1, params.M);
          self.q_ph =  np.ones((params.M, 1)) * (-1);
          self.L_ph =  [None]*params.M   #cell(1, params.M);
          self.Lpp_ph = [None]*(params.M + 1)  #cell(1, params.M + 1); # need an extra epoch here (osama)
          self.Y_ph =   [None]*params.M  #cell(1, params.M);
          self.P_MA_ph = [None]*params.M #cell(1, params.M);
      #------------------------------------------------------------
      #------------------------------------------------------------
      def compute_E_matrix(self, i, m_F):
            if (sum(i) == 0): # E matrix for only previous state faults
                self.E= np.zeros( (self.m, self.n_M + self.m) );
                self.E[:, end-self.m + 1:end)= eye(self.m];
            else # E matrix with faults in the PH
                self.E= zeros( self.m + m_F*length(i) , self.n_M + self.m );
                self.E( end-self.m+1 : end , end-self.m+1:end )= eye(self.m); # previous bias
                for j= 1:length(i)
                    self.E( m_F*(j-1)+1 : m_F*j , (i(j)-1)*m_F + 1 : i(j)*m_F )= eye(m_F); # landmark i(j) faulted
      
