import math
import numpy as np
import sys
from scipy.stats import norm
from scipy.stats import ncx2
from scipy.special import comb
from scipy.optimize import fminbound
from scipy.optimize import minimize

class IntegrityMonitoringClassFgExpOff:

       m= 3
       calculate_A_M_recursively = 0;
       C_req = None
       p_hmi = None
       detector_threshold = None
       is_extra_epoch_needed= -1 # initialize as (-1), then a boolean
       ind_im= [1,2,9];


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

       def __init__(self,params,estimator):

            # if the preceding horizon is fixed in epochs --> set M
            if (params.SWITCH_FIXED_LM_SIZE_PH == 1):
                self.M= 0;
            else:
                self.M= params.M;

            # continuity requirement
            self.C_req= params.continuity_requirement;

            # initialize the preceding horizon
            self.XX_ph=  [None]*params.preceding_horizon_size+1
            self.XX_ph[0]=  estimator.XX;
            self.D_bar_ph=  [None] * params.preceding_horizon_size
            self.PX_prior=  estimator.PX_prior;
            self.Gamma_prior= estimator.Gamma_prior;
            self.n_gps_ph= np.zeros((params.preceding_horizon_size, 1));
            self.H_gps_ph= [None] * params.preceding_horizon_size
            self.H_lidar_ph= [None]*params.preceding_horizon_size
            self.n_ph=    np.zeros((params.M, 1));
            self.Phi_ph=  [None]*(params.M + 1)
            self.H_ph=   [None] *params.M
            self.gamma_ph= [None]* params.M
            self.q_ph=     np.ones(params.M, 1) * (-1);
            self.L_ph=  [None]* params.M
            self.Lpp_ph=  [None]* (params.M+1)  # need an extra epoch here (osama)
            self.Y_ph=      [None]* params.M
            self.P_MA_ph=   [None]* params.M

       # ----------------------------------------------
       # ----------------------------------------------
       def optimization_fn(self, f_M_mag, fx_hat_dir, M_dir, sigma_hat, l, dof):
            neg_p_hmi= - ( (1 - norm.cdf(l , f_M_mag * fx_hat_dir, sigma_hat) +norm.cdf(-l , f_M_mag * fx_hat_dir, sigma_hat))*ncx2.cdf(self.T_d, dof, f_M_mag**2 * M_dir ) );
            return neg_p_hmi
       # ----------------------------------------------
       # ----------------------------------------------
       def compute_E_matrix_fg(self, i, m_F):
            if (sum(i) == 0): # E matrix for only previous state faults
                self.E = np.zeros((self.m, self.n_total))
                self.E[0:1, 0:1] = np.eye(2)
                self.E[3, 9] = 1
            else: # E matrix with a single LM fault
                fault_type_indicator = -1 * np.ones(i.size, 1)
                for j in range(1,i.shape[0]):
                    if (i[j] > self.n_L_M):
                        fault_type_indicator[j] = 3
                    else:
                        fault_type_indicator[j] = 1
                self.E = np.zeros(self.m + sum(fault_type_indicator*2), self.n_total)
                # previous bias
                self.E[0:1, 0:1] = np.eye(2)
                self.E[3, 9] = 1
                r_ind = self.m + 1
                for j in range(1,i.shape[0]):
                    if (fault_type_indicator[j] == 1):
                        ind = self.lidar_msmt_ind[:, i[j]]
                        self.E[r_ind: r_ind + m_F - 1, ind[:].transpose()] = np.eye(m_F)
                        r_ind = r_ind + m_F
                    else:
                        ind = self.gps_msmt_ind[:, i[j] - self.n_L_M]
                        self.E[r_ind: r_ind + 5, ind[:].transpose()] = np.eye(6)
                        r_ind = r_ind + 6
       # ----------------------------------------------
       # ----------------------------------------------
       def update_preceding_horizon(self, estimator):

           self.Phi_ph=   [math.inf, estimator.Phi_k, self.Phi_ph[2:self.M]]
           self.H_ph=     [estimator.H_k,   self.H_ph[0:self.M-1]]
           self.n_ph=     np.concatenate((estimator.n_k,self.n_ph[0:self.M-1]),axis = 1)
           self.XX_ph=    [estimator.XX,    self.XX_ph[0:self.M]]
           self.D_bar_ph= [self.inf, estimator.D_bar, self.D_bar_ph[1:self.M]]
           self.H_gps_ph=     [estimator.H_k_gps,   self.H_gps_ph[1:self.M-1]]
           self.H_lidar_ph=     [estimator.H_k_lidar,   self.H_lidar_ph[1:self.M-1]]
           self.n_gps_ph= [estimator.n_gps_k,   self.n_gps_ph[1:self.M-1]]
       # ----------------------------------------------
       # ----------------------------------------------
       def compute_hypotheses(self, params):
          # probability of "r" or more simultaneous faults
          flag_out= 0;
          for r in range(self.P_F_M.shape[0]):
              if  (np.sum(self.P_F_M)**r or math.factorial(r)  < params.I_H):
                  self.n_max= r-1;
                  flag_out= 1;
                  break

          # if no "r" holds --> all landmarks failing simultaneously must be monitored
          if (flag_out==0):
             self.n_max= r

          if (self.n_max > 1):
              print('n_max: #d\n', self.n_max)
              if (params.SWITCH_ONLY_ONE_LM_FAULT ==1):
                  self.n_max= 1;

          # compute number of hypotheses
          self.n_H= 0;
          self.inds_H= [None]*200
          start_ind= 1;
          for num_faults in range(self.n_max):
              if (params.SWITCH_FACTOR_GRAPHS==1 and (params.SWITCH_SIM==0)):
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
       # ----------------------------------------------
       # ----------------------------------------------
       def  build_state_of_interest_extraction_matrix(self, params, current_state):
            if (params.SWITCH_SIM == 1):
                # In the simulation mode
                alpha= np.concatenate(( np.zeros( (self.M * params.m, 1)),-math.sin(current_state[params.ind_yaw-1]),math.cos( current_state[params.ind_yaw-1])),axis = 1)
            else:
                # In the experiment mode
                alpha= np.concatenate((np.zeros( (self.M * (params.m), 1)),-math.sin( current_state[params.ind_yaw-1]),math.cos( current_state[params.ind_yaw-1]),np.zeros((params.m-2 , 1))),aixs = 1)

            return alpha
       # ----------------------------------------------
       # ----------------------------------------------

       def  compute_p_hmi_H(self, alpha, fault_ind, params):

            # build extraction matrix
            if (fault_ind == 0):
                self.compute_E_matrix_fg[ 0, params.m_F];
            else:
                self.compute_E_matrix_fg[ self.inds_H[fault_ind], params.m_F];

            # Worst-case fault direction
            f_M_dir= np.dot(np.dot(np.dot(np.dot(np.dot(np.transpose(self.E),np.inv(np.dot(np.dot(self.E,self.M_M),np.transpose(self.E)))),self.E),self.A),self.PX_M),alpha);
            f_M_dir= np.dot(f_M_dir,np.ing(np.norm(f_M_dir))); # normalize

            # worst-case fault magnitude
            fx_hat_dir= np.abs( np.dot(np.dot(np.dot(np.transpose(alpha),np.inv( self.Gamma_fg)),np.transpose(self.A)),f_M_dir) );
            M_dir= np.abs( np.dot(np.dot(np.transpose(f_M_dir),self.M_M),f_M_dir) );

            # save the interesting values for the optimization
            self.counter_H= self.counter_H + 1;
            self.noncentral_dof[self.counter_H-1]=  self.n_M + self.n_M_gps;
            self.f_dir_sig2[self.counter_H-1]= (np.dot(fx_hat_dir,np.inv(self.sigma_hat)))**2;
            self.M_dir[self.counter_H-1]= M_dir;
            # worst-case fault magnitude
            f_mag_min = 0
            f_mag_max = 5
            f_mag_inc = 5
            p_hmi_H_prev = -1
            for k in range(1, 11):
                args = np.c_[self.f_M_mag, fx_hat_dir, M_dir, self.sigma_hat, params.alert_limit, self.n_M]
                bound = np.c_[f_mag_min, f_mag_max]
                [f_M_mag_out, p_hmi_H] = minimize(self.optimization_fn,args,bound)

                # make it a positive number
                p_hmi_H = -p_hmi_H

                # check if the new P(HMI|H) is smaller
                if (k == 1 or p_hmi_H_prev < p_hmi_H):
                    p_hmi_H_prev = p_hmi_H
                    f_mag_min = f_mag_min + f_mag_inc
                    f_mag_max = f_mag_max + f_mag_inc
                else:
                    p_hmi_H = p_hmi_H_prev
                    break
                return p_hmi_H
            if np.abs(self.optimization_fn(0, fx_hat_dir, M_dir, self.sigma_hat, params.alert_limit, self.n_M + self.n_M_gps)) > 1e-10 :
                f_mag_min = 0
                f_mag_max = 5
                f_mag_inc = 5
                p_hmi_H_prev = -1
                for k in range(1, 11):
                    args = np.c_[self.f_M_mag, fx_hat_dir, M_dir, self.sigma_hat, params.alert_limit, self.n_M]
                    bound = np.c_[f_mag_min, f_mag_max]
                    [f_M_mag_out, p_hmi_H] = minimize(self.optimization_fn, args, bound)

                    # make it a positive number
                    p_hmi_H = -p_hmi_H

                    # check if the new P(HMI|H) is smaller
                    if (k == 1 or p_hmi_H_prev < p_hmi_H):
                        p_hmi_H_prev = p_hmi_H
                        f_mag_min = f_mag_min + f_mag_inc
                        f_mag_max = f_mag_max + f_mag_inc
                    else:
                        p_hmi_H = p_hmi_H_prev
                        break
            else:
                p_hmi_H_1 = 0
                p_hmi_H_2 = 0
                p_hmi_H_3 = 0
                p_hmi_H_4 = 0
                [f_M_mag_out, p_hmi_H_1] = fminbound(self.optimization_fn(self.f_M_mag, fx_hat_dir, M_dir, self.sigma_hat, params.alert_limit,self.n_M + self.n_M_gps), 0, 10)
                p_hmi_H_1 = -p_hmi_H_1
                if (p_hmi_H_1 < 1e-10) or (f_M_mag_out > 8):
                    [f_M_mag_out, p_hmi_H_2] = fminbound(self.optimization_fn(self.f_M_mag, fx_hat_dir, M_dir, self.sigma_hat, params.alert_limit,self.n_M + self.n_M_gps), 10, 100)
                    p_hmi_H_2 = -p_hmi_H_2
                    if (p_hmi_H_2 < 1e-10) or (f_M_mag_out > 180):
                        [f_M_mag_out, p_hmi_H_3] = fminbound(self.optimization_fn(self.f_M_mag, fx_hat_dir, M_dir, self.sigma_hat, params.alert_limit,self.n_M + self.n_M_gps), 100, 1000)
                        p_hmi_H_3 = -p_hmi_H_3

                p_hmi_H = max(p_hmi_H_1, p_hmi_H_2, p_hmi_H_3, p_hmi_H_4)

       # ----------------------------------------------
       # ----------------------------------------------
       def compute_required_epochs_for_min_LMs(self, params, estimator):

           self.n_M= estimator.n_k;
           i= 0; # initialize i to zero to indicate the current epoch
           if (np.sum(self.n_ph) != 0):
               for i in range(1,self.n_ph.shape[0]):
                   self.n_M= self.n_M + self.n_ph[i-1];
                   # if the preceding horizon is long enough --> stop
                   if ((self.n_M/params.m_F)  >= params.min_n_L_M ):
                      break

           # set the variables
           self.n_L_M= self.n_M / params.m_F;
           estimator.n_L_M= self.n_L_M;
           self.M= i + 1; # preceding epochs plus the current
       # ----------------------------------------------
       # ----------------------------------------------
       def compute_whiten_jacobian_A(self, estimator, params):
           # this function computes the A jacobian for future use in integrity monitoring

           # initialize normalized Jacobian
           self.A= np.zeros( (self.n_total, self.m_M) );

           # indices of the absolute measurements in the rows of matrix A
           self.lidar_msmt_ind = []
           self.gps_msmt_ind = []

           # plug the prior in A
           self.A[ 0:params.m, 0:params.m ] = np.sqrtm( self.Gamma_prior )

           # pointers to the next part of A to be filled
           r_ind= params.m + 1;
           c_ind= 1;

           # build A whithen Jacobian
           for i in range( self.M-2 , -1 , 0):

               if i == 0:
                   # Whiten IMU model and then plug it in A
                   self.A[ r_ind : r_ind + params.m - 1, c_ind : c_ind + params.m - 1]= np.sqrtm( np.linalg.inv(estimator.D_bar) ) * estimator.Phi_k
                   self.A[ r_ind : r_ind + params.m - 1, c_ind + params.m : c_ind + 2*( params.m) - 1]= -np.sqrtm(np.linalg.inv(estimator.D_bar) )

                   # update the row & column indices
                   r_ind = r_ind + params.m
                   c_ind = c_ind + params.m

                   # plug the whitened gps model in A
                   if estimator.n_gps_k != 0:
                        self.A[ r_ind : r_ind + estimator.n_gps_k - 1, c_ind : c_ind + params.m - 1 ] = estimator.H_k_gps

                        # record gps msmt indieces in A
                        self.gps_msmt_ind = [self.gps_msmt_ind, np.arrange(r_ind, r_ind + estimator.n_gps_k - 1).transpose() ]

                        # update the row indices
                        r_ind = r_ind + estimator.n_gps_k


                   # plug the whitened lidar model in A
                   self.A[r_ind : r_ind + estimator.n_k - 1, c_ind : c_ind + params.m - 1 ] = estimator.H_k_lidar

                   # record lidar msmt indieces in A
                   self.lidar_msmt_ind = [ self.lidar_msmt_ind, np.reshape( np.arrange(r_ind, r_ind + estimator.n_k - 1), (params.m_F , []) ) ];

               else:
                   # Whiten IMU model and then plug it in A
                   self.A[ r_ind : r_ind + params.m - 1, c_ind : c_ind + params.m - 1] = np.sqrtm( np.linalg.inv(self.D_bar_ph[ i + 1 ]) ) * self.Phi_ph[ i + 1 ]
                   self.A[ r_ind : r_ind + params.m - 1, c_ind + params.m : c_ind + 2*(params.m) - 1] = -np.sqrtm( np.linalg.inv(self.D_bar_ph[ i + 1 ]) )

                   # update the row & column indices
                   r_ind = r_ind + params.m
                   c_ind = c_ind + params.m

                   # plug the whitened gps model in A
                   if (self.n_gps_ph( i ) != 0):
                        self.A[  r_ind : r_ind + self.n_gps_ph[ i ] - 1, c_ind : c_ind + params.m - 1 ] = self.H_gps_ph[i]

                        # record gps msmt indieces in A
                        self.gps_msmt_ind = [self.gps_msmt_ind, np.arrange(r_ind,  r_ind + self.n_gps_ph[i] - 1).transpose()]

                        # update the row indices
                        r_ind = r_ind + self.n_gps_ph[i]

                   # plug the whitened lidar model in A
                   n_L_i= self.n_ph[ i ] / params.m_F
                   self.A[ r_ind : r_ind + self.n_ph[i] - 1, c_ind : c_ind + params.m - 1 ]= self.H_lidar_ph[ i ]

                   # record lidar msmt indieces in A
                   self.lidar_msmt_ind = [ self.lidar_msmt_ind, np.reshape( np.arrange(r_ind, r_ind + self.n_ph(i) - 1), (params.m_F , [] )) ]

                   # update the row index
                   r_ind = r_ind + self.n_ph[i];
