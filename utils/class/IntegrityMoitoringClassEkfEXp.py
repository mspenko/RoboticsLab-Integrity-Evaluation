import numpy as np
import math
from scipy.stats import norm
from scipy.stats.distributions import chi2
from scipy.optimize import fminbound
from scipy.optimize import minimize
from scipy.special import comb
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
              if (self.n_L_M - self.n_max < 2): # need at least 5 msmts (3 landmarks) to monitor one landmark fault
                 print('Not enough redundancy: n_L_M = #d, n_max = #d\n', self.n_L_M, self.n_max)
                 self.p_hmi= 1;
            
              else: # if we don't have enough landmarks --> P(HMI)= 1   
                    self.P_H= np.ones(self.n_H, 1) * math.inf; # initializing P_H vector
                    for i in range(self.n_H.shape[0]):
                        # build extraction matrix
                        if (i == 0):
                           self.compute_E_matrix(0, params.m_F);
                        else:
                           self.compute_E_matrix(self.inds_H[i], params.m_F);
                        # Worst-case fault direction
                        f_M_dir= np.dot(np.dot(np.dot(np.dot(np.transpose(self.E),np.inv(self.E * self.M_M * np.transpose(self.E))),self.E), np.transpose(self.A_M)), alpha);
                        f_M_dir= np.dot(f_M_dir,np.inv(np.linalg.norm(f_M_dir))); # normalize
                
                        # worst-case fault magnitude
                        fx_hat_dir= np.dot(np.dot(np.transpose(alpha),self.A_M),f_M_dir);
                        M_dir= np.dot(np.dot(np.transpose(f_M_dir),self.M_M),f_M_dir);
                
                        # worst-case fault magnitude
                        f_mag_min= 0;
                        f_mag_max= 5;
                        f_mag_inc= 5;
                        p_hmi_H_prev= -1;
                        for k in range(11):
                            args = np.c_[f_M_mag, fx_hat_dir, M_dir, self.sigma_hat, params.alert_limit, np.dot(params.m_F,self.n_L_M)]
                            bound = np.c_[f_mag_min, f_mag_max]
                            [f_M_mag_out, p_hmi_H]= minimize(self.optimization_fn,args,bound);
                            # make it a positive number
                            p_hmi_H= -p_hmi_H;
                    
                            # check if the new P(HMI|H) is smaller
                            if (k == 1 or p_hmi_H_prev < p_hmi_H):
                               p_hmi_H_prev= p_hmi_H;
                               f_mag_min= f_mag_min + f_mag_inc;
                               f_mag_max= f_mag_max + f_mag_inc;
                            else:
                               p_hmi_H= p_hmi_H_prev;

                        # Add P(HMI | H) to the integrity risk
                        if (i == 0):
                           self.P_H_0= prod( 1 - self.P_F_M );
                           self.p_hmi= self.p_hmi + np.dot(p_hmi_H,self.P_H_0);
                        else:
                           # unfaulted_inds= all( 1:self.n_L_M ~= fault_inds(i,:)', 1 );
                           self.P_H[i]= np.prod( self.P_F_M( self.inds_H[i] ) ); #...
                           # * prod( 1 - P_F_M(unfaulted_inds)  );
                           self.p_hmi= self.p_hmi + p_hmi_H * self.P_H(i)
              # store integrity related data
              data.store_integrity_data(self, estimator, counters, params)
          #hey
          elif (counters.k_im > 1): # if it's the first time --> cannot compute Lpp_k 
               if (estimator.n_k == 0):
                 self.Lpp_k= self.Phi_ph[0];
               else:
                 self.Lpp_k= self.Phi_ph[0] - np.dot(np.dot(self.L_k,self.H_k),self.Phi_ph[0]);
    
               if (params.SWITCH_FIXED_LM_SIZE_PH == 1):
                  self.M = self.M + 1;
                  if (self.is_extra_epoch_needed == 1):
                     self.is_extra_epoch_needed= 0;
          else: # first time we get lidar msmts
               self.Lpp_k= 0;
               if (params.SWITCH_FIXED_LM_SIZE_PH==1):
                  if (self.is_extra_epoch_needed == 1):
                     self.is_extra_epoch_needed= 0;
       

          # store time
          data.im.time[counters.k_im]= counters.time_sim;

          # update the preceding horizon 
          update_preceding_horizon(self, estimator, params) 
      #--------------------------------------------------------------------------
      #--------------------------------------------------------------------------
      def compute_A_M_matrix(selfestimator):
          # build matrix A_M in the first time
          if (isempty(self.A_M) or self.calculate_A_M_recursively==0): 
              # allocate L_k and initialize
              self.A_M= np.zeros((self.m, self.n_M + self.m));
              self.A_M[:,1:estimator.n_k]= self.L_k;
    
              for i in range(self.M.shape[0]):
                  if (i == 1):
                     Dummy_Variable= self.Lpp_k;
                  else:
                     Dummy_Variable= np.dot(Dummy_Variable,self.Lpp_ph[i-1]);
                  # if no landmarks in the FoV at some time in the preceding horizon
                  if (self.n_ph(i) > 0):
                     n_start= estimator.n_k + np.sum( self.n_ph[1:i] ) + 1;
                     n_end=   estimator.n_k + np.sum( self.n_ph[1:i+1] );
                     self.A_M[:,n_start : n_end]= Dummy_Variable * self.L_ph[i];
    
              # last entry with only PSIs
              self.A_M[:, self.n_M+1 : self.n_M + self.m] = np.dot(Dummy_Variable,self.Lpp_ph[self.M]);
          # calculate matrix A_M recusively
          else:   
             self.A_M=np.array([self.L_k, self.Lpp_k*self.A_M]);
             self.A_M[:, self.n_M +1 : end-self.m] = None;
             self.A_M[:, self.A_M[self.A_M.shape[0]]-self.m+1 : self.A_M[self.A_M.shape[0]]] = np.dot(self.A_M[:, self.A_M[self.A_M.shape[0]]-self.m+1 : self.A_M[self.A_M.shape[0]]],np.inv(self.Lpp_ph[self.M +1]));
      #--------------------------------------------------------------------------
      #--------------------------------------------------------------------------
      def compute_B_bar_matrix(self, estimator):
          # Augmented B
          self.B_bar= math.inf*np.ones( self.n_M , self.n_M + self.m );
          A_prev= np.dot(np.inv(self.Lpp_k),self.A_M[ : , estimator.n_k + 1:self.A_M[self.A_M.shape[0]] ]);
          B_ind_row_start= estimator.n_k + 1;
          B_ind_col_end= estimator.n_k;

          # accounting for the case where there are no landmarks in the FoV at epoch k
          if (estimator.n_k > 0):
             self.B_bar[1:estimator.n_k , :]=np.concatenate(( eye(estimator.n_k), np.dot(np.dot(-self.H_k,self.Phi_ph[1]), A_prev) ),axis = 1);

          # Recursive computation of B
          for i in range(self.M,shape[0]):
              A_prev= np.dot(inv(self.Lpp_ph[i]),A_prev[:, self.n_ph(i)+1:end]);
          # accounting for the case where there are no landmarks in the FoV at
          # one of the epochs in the preceding horizon
          if (self.n_ph[i] > 0):
             B= np.concatenate((np.eye( self.n_ph[i] ) , np.dot(np.dot(-self.H_ph[i],self.Phi_ph[i+1]),A_prev)),axis = 1);
             B_ind_row_end= B_ind_row_start + self.n_ph(i) - 1;
             self.B_bar[B_ind_row_start:B_ind_row_end, 1:B_ind_col_end]= 0;
             self.B_bar[B_ind_row_start:B_ind_row_end, B_ind_col_end+1:end]= B;

             # increase row index for next element B
             B_ind_row_start= B_ind_row_start + self.n_ph[i];
             B_ind_col_end= B_ind_col_end + self.n_ph[i];
      #--------------------------------------------------------------------------
      #--------------------------------------------------------------------------
      def compute_hypotheses(self, params):
          # probability of "r" or more simultaneous faults
          flag_out= 0;
          for r in range(self.P_F_M.shape[0]):
              if  (np.sum(self.P_F_M)**r or factorial[r]  < params.I_H):
                  self.n_max= r-1;
                  flag_out= 1;
                  break

          # if no "r" holds --> all landmarks failing simultaneously must be monitored
          if (flag_out==0): 
             self.n_max= r

          if (self.n_max > 1):
              printf('n_max: #d\n', self.n_max);
              if (params.SWITCH_ONLY_ONE_LM_FAULT ==1):
                  self.n_max= 1;

          # compute number of hypotheses
          self.n_H= 0;
          self.inds_H= [None]*200
          start_ind= 1;
          for num_faults in range(self.n_max):
              if (params.SWITCH_FACTOR_GRAPHS and (params.SWITCH_SIM==0)):
                  new_H= comb(self.n_L_M + (self.n_M_gps/6), num_faults);
                  self.n_H= self.n_H + new_H;
                  self.inds_H[ start_ind:start_ind+new_H - 1, 1]=np.array([ comb( self.n_L_M + (self.n_M_gps/6), num_faults)]).reshape(2,( self.n_L_M + (self.n_M_gps/6)).shape[0],num_faults.shape[0]);
                  start_ind= start_ind + new_H;
              else:
                  new_H= comb(self.n_L_M, num_faults);
                  self.n_H= self.n_H + new_H;
                  self.inds_H[ start_ind:start_ind+new_H - 1, 1 ]=...
                  np.array([ comb(self.n_L_M, num_faults)]) .reshape(2,self.n_L_M.shape[0],num_faults.shape[0])
                  start_ind= start_ind + new_H;
          self.inds_H[start_ind:self.inds_H[self.inds_H.shape[0]]]= None;

      #--------------------------------------------------------------------------
      #--------------------------------------------------------------------------
      def compute_Y_M_matrix(self, estimator):
          # if it's the first epoch --> build the Y_M
          if isempty(self.Y_M):
             self.Y_M= np.zeros( (self.n_M, self.n_M) );
             self.Y_M[ 1:estimator.n_k, 1:estimator.n_k ]= estimator.Y_k;
             for i in range(self.M):
                 n_start= estimator.n_k + np.sum( self.n_ph[1:i-1] ) + 1;
                 n_end  = estimator.n_k + np.sum( self.n_ph[1:i] );
                 self.Y_M[ n_start: self.Y_M[self.Y_M.shape[0]] , n_start:self.Y_M[self.Y_M.shape[0]] ]= self.Y_ph[i];
          else: # update Y_M
            self.Y_M=np.concatenate((np.concatenate((estimator.Y_k, np.zeros((estimator.n_k,np.sum(self.n_ph[1:self.M])))),axis = 1),np.concatenate((np.zeros((np.sum(self.n_ph[1:self.M]),estimator.n_k)), self.Y_M[1:np.sum(self.n_ph[1:self.M]), 1:np.sum(self.n_ph[1:self.M])]),axis =1)),axis = 0)
      #--------------------------------------------------------------------------
      #--------------------------------------------------------------------------
      def update_preceding_horizon(self, estimator, params):
            
            # TODO: organize 
                if (params.SWITCH_FIXED_LM_SIZE_PH == 1):
                    self.n_ph=     np.concatenate((estimator.n_k,self.n_ph[1:self.M]),axis = 0);
                    self.gamma_ph= np.concatenate((estimator.gamma_k,self.gamma_ph[1:self.M]),axis = 0);
                    self.q_ph=     np.concatenate((estimator.q_k,self.q_ph[1:self.M]),axis = 0);
                    self.Phi_ph=   np.concatenate((self.Phi_k, self.Phi_ph[1:self.M+ 1]),axis = 1); ######## CAREFUL
                    self.H_ph=     np.concatenate((self.H_k,self.H_ph[1:self.M]),axis = 1);
                    self.L_ph=     np.concatenate((self.L_k,self.L_ph[1:self.M]),axis = 1);
                    self.Lpp_ph=   np.concatenate((self.Lpp_k,self.Lpp_ph[1:self.M]),axis = 1);
                    self.Y_ph=     np.concatenate((estimator.Y_k,self.Y_ph[1:self.M]),axis = 1);
                    self.P_MA_ph=  np.concatenate((self.P_MA_k,self.P_MA_ph[1:self.M]),axis = 1);

                else:
                    self.n_ph=     np.concatenate((estimator.n_k,self.n_ph[1:self.n_ph[self.n_ph.shape[0]]-1]),axis = 0);
                    self.gamma_ph= np.concatenate((estimator.gamma_k, self.gamma_ph[1:self.gamma_ph[self.gamma_ph.shape[0]]-1]),axis = 1);
                    self.q_ph=     np.concatenate((estimator.q_k,self.q_ph[1:self.q_ph[self.q_ph.shape[0]]]-1),axis = 0);
                    self.Phi_ph=   np.concatenate((self.Phi_k,self.Phi_ph[1:self.Phi_ph[self.Phi_ph.shape[0]]-1]),axis = 0); ######## CAREFUL
                    self.H_ph=     np.concatenate((self.H_k,self.H_ph[1:self.H_ph[self.H_ph.shape[0]]-1]),axis = 0);
                    self.L_ph=     np.concatenate((self.L_k,self.L_ph[1:self.L_ph[self.L_ph.shape[0]]-1]),axis = 0);
                    self.Lpp_ph=   np.concatenate((self.Lpp_k,self.Lpp_ph[1:self.Lpp_ph[self.Lpp_ph.shape[0]]-1]),axis  = 0);
                    self.Y_ph=     np.concatenate((estimator.Y_k,self.Y_ph[1:self.Y_ph[self.Y_ph.shape[0]]-1]),axis =0);
                    self.P_MA_ph=  np.concatenate((self.P_MA_k,self.P_MA_ph[1:self.P_MA_p[self.P_MA_p.shape[0]]-1]),axis = 0);

