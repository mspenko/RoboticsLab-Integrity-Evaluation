import numpy as np
import math
from scipy.stats import norm
from scipy.stats.distributions import chi2

class IntegrityMonitoringCLassEkfExp:
      m= 3
      calculate_A_M_recursively = 0
      C_req = None
      p_hmi = None
      detector_threshold = None
        
      is_extra_epoch_needed= -1 # initialize as (-1), then a boolean
      ind_im= np.array([1,2,9]);

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
      m_M  = None      # number of states to estimate
      n_total  = None  # total number of msmts (prior + relative + abs)
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
        
      noncentral_dof= [None]*10000 #cell(10000,1)
      f_dir_sig2= [None]*10000 #cell(10000,1)
      M_dir= [None]*10000 #cell(10000,1)
      counter_H=0
      #======================================================================================================
      def __init__(self,params, estimator):            
         # if the preceding horizon is fixed in epochs --> set M
         if (params.SWITCH_FIXED_LM_SIZE_PH == 1):
             self.M= 0;
         else:
             self.M= params.M;
            
         # continuity requirement
         self.C_req= params.continuity_requirement;
            
         # initialize the preceding horizon
         self.n_ph=     np.zeros( (params.M, 1) );
         self.Phi_ph=   [None]*( params.M + 1) #np.cell( 1, params.M + 1 ); # need an extra epoch here
         self.H_ph=     [None]*( params.M) #np.cell( 1, params.M );
         self.gamma_ph= [None]*( params.M) #np.cell(1, params.M);
         self.q_ph=     np.ones(params.M, 1) * (-1);
         self.L_ph=     [None]*( params.M) #np.cell(1, params.M);
         self.Lpp_ph=   [None]*( params.M + 1) #np.cell(1, params.M + 1); # need an extra epoch here (osama)
         self.Y_ph=     [None]*( params.M) #np.cell(1, params.M);
         self.P_MA_ph=  [None]*( params.M) #np.cell(1, params.M);
      #--------------------------------------------------------------------------
      #--------------------------------------------------------------------------
      def optimization_fn(self, f_M_mag, fx_hat_dir, M_dir, sigma_hat, l, dof):
          neg_p_hmi= - ( (1 - norm.cdf(l , f_M_mag * fx_hat_dir, sigma_hat) +norm.cdf(-l , f_M_mag * fx_hat_dir, sigma_hat))* ncx2cdf(self.T_d, dof, f_M_mag**2 * M_dir ) )      
          return neg_p_hmi
      #--------------------------------------------------------------------------
      #--------------------------------------------------------------------------
      def compute_E_matrix(self, i, m_F):
            if (sum(i) == 0): # E matrix for only previous state faults
                self.E= np.zeros((self.m, self.n_M + self.m));
                self.E[:, self.E[self.E.shape[0]]-self.m + 1:self.E[self.E.shape[0]]]= np.eye(self.m);
            else: # E matrix with faults in the PH
                self.E= np.zeros((self.m + m_F*i.shape[0] , self.n_M + self.m));
                self.E[ self.E[self.E.shape[0]]-self.m+1 :self.E[self.E.shape[0]] , end-self.m+1:self.E[self.E.shape[0]] ]= np.eye(self.m); # previous bias
                for j in range(i.shape[0]):
                    self.E[ m_F*[j-1]+1 : m_F*j , (i[j]-1)*m_F + 1 : i[j]*m_F ]= np.eye(m_F); # landmark i(j) faulted
      #--------------------------------------------------------------------------
      #--------------------------------------------------------------------------
      def monitor_integrity(self, estimator, counters, data, params):

          # keep only the elements for the [x-y-theta]
          # the state evolution matrix from one lidar msmt to the next one
          self.Phi_k= estimator.Phi_k**12;  ######## CAREFUL
          self.Phi_k= self.Phi_k[ params.ind_pose, params.ind_pose ]; 

          # build L and H for the current time using only the pose indexes
          if (estimator.n_k == 0):# no landmarks in the FoV at epoch k
             self.H_k= None
             self.L_k= None
          else: # extract the indexes from the pose
             self.H_k= estimator.H_k[:, params.ind_pose];
             self.L_k= estimator.L_k[params.ind_pose, :];

          # current horizon measurements
          self.n_M= np.sum( self.n_ph ) + estimator.n_k;
          self.n_L_M= np.dot(self.n_M, np.inv(params.m_F));

           # the first time we have enough preceding horizon
          if (self.is_extra_epoch_needed == -1 and self.n_L_M >= params.min_n_L_M and counters.k_im > 2):
                self.is_extra_epoch_needed= 1;
            # monitor integrity if the number of LMs in the preceding horizon is more than threshold
          if((params.SWITCH_FIXED_LM_SIZE_PH and self.n_L_M >= params.min_n_L_M and self.is_extra_epoch_needed == false ) or( params.SWITCH_FIXED_LM_SIZE_PH==0 and counters.k_im > self.M + 2 )):
          
              # Modify preceding horizon to have enough landmarks
              if (params.SWITCH_FIXED_LM_SIZE_PH ==1):
                self.n_M= estimator.n_k;
                for i in range(self.n_ph.shape[0]):
                    self.n_M= self.n_M + self.n_ph[i];
                    # if the preceding horizon is long enough --> stop
                    if (self.n_M >= params.min_n_L_M * params.m_F):
                        break
                # set the variables
                self.n_L_M= np.dot(self.n_M,np.inv(params.m_F));
                self.M= i;
              # common parameters
              alpha= np.array([[-math.sin(estimator.XX[params.ind_yaw])],[np.cos(estimator.XX[params.ind_yaw])], [0]]);
              self.sigma_hat= np.sqrt( np.dot(np.dot(np.transpose(alpha),estimator.PX(params.ind_pose, params.ind_pose)),alpha) );
              # detector threshold
              self.T_d = np.sqrt( chi2.ppf( 1 - params.continuity_requirement , self.n_M ) );
    
              # If there are no landmarks in the FoV at k 
              if (estimator.n_k == 0):
                 self.Lpp_k= self.Phi_ph[1];
              else:
                 self.Lpp_k= self.Phi_ph[1] - np.dot(np.dot(self.L_k, self.H_k), self.Phi_ph[0]);
              # accounting for the case where there are no landmarks in the FoV at
              # epoch k and the whole preceding horizon
              if (self.n_M == 0):
                 self.Y_M=   None
                 self.A_M=   None
                 self.B_bar= None
                 self.q_M= 0
                 self.detector_threshold= 0
                 self.p_hmi= 1
              else:
                 # Update the innovation vector covarience matrix for the new PH
                 self.compute_Y_M_matrix(estimator)

              # compute the A matrix for the preceding horizon
              self.compute_A_M_matrix(estimator)

              # compute B_bar matrix
              self.compute_B_bar_matrix(estimator)    

              # M matrix
              self.M_M= np.dot(np.dot(np.transpose(self.B_bar),np.inv(self.Y_M)), self.B_bar);

              # set the threshold from the continuity req
              self.detector_threshold= chi2.ppf(1 - self.C_req, self.n_M);
        
              # compute detector
              self.q_M= np.sum(self.q_ph[1:self.M]) + estimator.q_k;
        
              # TODO: very inefficient --> do not transform from cell to matrix
              self.P_MA_M = np.array([ [self.P_MA_k], [np.array(np.transpose(self.P_MA_ph[1:self.M]))]])
        
              # fault probability of each association in the preceding horizon
              self.P_F_M= self.P_MA_M + params.P_UA;

              # compute the hypotheses (n_H, n_max, inds_H)
              self.compute_hypotheses(params)
              # initialization of p_hmi
              self.p_hmi= 0
              if (obj.n_L_M - obj.n_max < 2): # need at least 5 msmts (3 landmarks) to monitor one landmark fault
                 printf('Not enough redundancy: n_L_M = %d, n_max = %d\n', obj.n_L_M, obj.n_max)
                 obj.p_hmi= 1;
            
              else % if we don't have enough landmarks --> P(HMI)= 1   
                   obj.P_H= ones(obj.n_H, 1) * inf; % initializing P_H vector
                   for i= 0:obj.n_H
                   % build extraction matrix
                   if i == 0
                      obj.compute_E_matrix(0, params.m_F);
                   else
                      obj.compute_E_matrix(obj.inds_H{i}, params.m_F);

