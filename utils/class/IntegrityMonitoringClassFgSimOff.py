import math
import numpy as np
import sys
from scipy.stats import norm
from scipy.stats import ncx2
from scipy.special import comb
from scipy.optimize import fminbound
from scipy.optimize import minimize

class IntegrityMonitoringClassFgSimOff:

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
            
            # if it's a simulation --> change the indexes
            if (params.SWITCH_SIM==1):
                self.ind_im= self.ind_im[0:3]
            
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
       def optimization_fn(obj, f_M_mag, fx_hat_dir, M_dir, sigma_hat, l, dof):
            neg_p_hmi= - ( (1 - norm.cdf(l , f_M_mag * fx_hat_dir, sigma_hat) +norm.cdf(-l , f_M_mag * fx_hat_dir, sigma_hat))*ncx2.cdf(obj.T_d, dof, f_M_mag**2 * M_dir ) );
            return neg_p_hmi
       # ----------------------------------------------
       # ----------------------------------------------
       def compute_E_matrix_fg(obj, i, m_F):
            if (sum(i) == 0): # E matrix for only previous state faults
                obj.E= np.zeros((obj.m, obj.n_total));
                obj.E[:, 0:obj.m]= np.eye(obj.m);
            else: # E matrix with a single LM fault
                obj.E= np.zeros((obj.m + m_F*i.shape[0] , obj.n_total));
                obj.E[ 0:obj.m , 0:obj.m ]= np.eye(obj.m); # previous bias
                for j in range(1,i.shape[0]):
                    ind= obj.abs_msmt_ind[:,i[j-1]];
                    obj.E[ obj.m + 1 + m_F*(j-1) : obj.m + m_F*(j) , np.transpose(ind) ]= np.eye[m_F]; # landmark i faulted
       # ----------------------------------------------
       # ----------------------------------------------
       def update_preceding_horizon(obj, estimator):
            
           obj.Phi_ph=   [math.inf, estimator.Phi_k, obj.Phi_ph[2:obj.M]];
           obj.H_ph=     [estimator.H_k,   obj.H_ph[0:obj.M-1]];
           obj.n_ph=     np.concatenate((estimator.n_k,obj.n_ph[0:obj.M-1]),axis = 1);
           obj.XX_ph=    [estimator.XX,    obj.XX_ph[0:obj.M]];
           obj.D_bar_ph= [inf, estimator.D_bar, obj.D_bar_ph[1:obj.M]];
       # ----------------------------------------------
       # ----------------------------------------------
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
       def  build_state_of_interest_extraction_matrix(obj, params, current_state):
            if (params.SWITCH_SIM == 1):
                # In the simulation mode
                alpha= np.concatenate(( np.zeros( (obj.M * params.m, 1)),-math.sin(current_state[params.ind_yaw-1]),math.cos( current_state[params.ind_yaw-1])),axis = 1) 
            else:
                # In the experiment mode
                alpha= np.concatenate((np.zeros( (obj.M * (params.m), 1)),-math.sin( current_state[params.ind_yaw-1]),math.cos( current_state[params.ind_yaw-1]),np.zeros((params.m-2 , 1))),aixs = 1) 

            return alpha
       # ----------------------------------------------
       # ----------------------------------------------

       def  compute_p_hmi_H(obj, alpha, fault_ind, params):

            # build extraction matrix
            if (fault_ind == 0):
                obj.compute_E_matrix_fg[ 0, params.m_F];
            else:
                obj.compute_E_matrix_fg[ obj.inds_H[fault_ind], params.m_F];

            # Worst-case fault direction
            f_M_dir= np.dot(np.dot(np.dot(np.dot(np.dot(np.transpose(obj.E),np.inv(np.dot(np.dot(obj.E,obj.M_M),np.transpose(obj.E)))),obj.E),obj.A),obj.PX_M),alpha);
            f_M_dir= np.dot(f_M_dir,no.ing(np.norm(f_M_dir))); # normalize

            # worst-case fault magnitude
            fx_hat_dir= np.abs( np.dot(np.dot(np.dot(np.transpose(alpha),np.inv( obj.Gamma_fg)),np.transpose(obj.A)),f_M_dir) );
            M_dir= np.abs( np.dot(np.dot(np.transpose(f_M_dir),obj.M_M),f_M_dir) );

            # save the interesting values for the optimization
            obj.counter_H= obj.counter_H + 1;
            obj.noncentral_dof[obj.counter_H-1]=  obj.n_M + obj.n_M_gps;
            obj.f_dir_sig2[obj.counter_H-1]= (np.dot(fx_hat_dir,np.inv(obj.sigma_hat)))**2;
            obj.M_dir[obj.counter_H-1]= M_dir;
            # worst-case fault magnitude
            f_mag_min= 0;
            f_mag_max= 5;
            f_mag_inc= 5;
            p_hmi_H_prev= -1;
            for k in range(1,11):
                args = np.c_[f_M_mag, fx_hat_dir, M_dir, obj.sigma_hat, params.alert_limit, obj.n_M]
                bound = np.c_[f_mag_min, f_mag_max]
                [f_M_mag_out, p_hmi_H] = minimize(obj.optimization_fn,args,bound)

                # make it a positive number
                p_hmi_H= -p_hmi_H;

                # check if the new P(HMI|H) is smaller
                if (k == 1 or p_hmi_H_prev < p_hmi_H):
                    p_hmi_H_prev= p_hmi_H;
                    f_mag_min= f_mag_min + f_mag_inc;
                    f_mag_max= f_mag_max + f_mag_inc;
                else:
                    p_hmi_H= p_hmi_H_prev;
                    break
                return p_hmi_H
       # ----------------------------------------------
       # ----------------------------------------------
       def compute_required_epochs_for_min_LMs(obj, params, estimator):

           obj.n_M= estimator.n_k;
           i= 0; # initialize i to zero to indicate the current epoch
           if (np.sum(obj.n_ph) != 0):
               for i in range(1,obj.n_ph.shape[0]):
                   obj.n_M= obj.n_M + obj.n_ph[i-1];
                   # if the preceding horizon is long enough --> stop
                   if ((obj.n_M/params.m_F)  >= params.min_n_L_M ):
                      break

           # set the variables
           obj.n_L_M= obj.n_M / params.m_F;
           estimator.n_L_M= obj.n_L_M;
           obj.M= i + 1; # preceding epochs plus the current
       # ----------------------------------------------
       # ----------------------------------------------
       def compute_whiten_jacobian_A(obj, estimator, params):
           # this function computes the A jacobian for future use in integrity monitoring

           # initialize normalized Jacobian
           obj.A= np.zeros( (obj.n_total, obj.m_M) );

           # indices of the absolute measurements in the rows of matrix A
           obj.abs_msmt_ind= None

           # plug the prior in A
           obj.A[ 0:params.m, 0:params.m ]= np.sqrtm( obj.Gamma_prior );

           # pointers to the next part of A to be filled
           r_ind= params.m + 1;
           c_ind= 1;

           # build A whithen Jacobian
           for i in range( obj.M-2 , -1 , 0):
    
               # gyro msmt submatrix
               obj.A[ r_ind, c_ind-1 : c_ind + params.m - 1 ]= np.dot(-params.sig_gyro_z**(-1),np.array([0, 0, 1/params.dt_sim]))
    
               obj.A[ r_ind, c_ind + params.m : c_ind + 2*params.m - 1 ]= np.dot(params.sig_gyro_z^(-1),np.array([0, 0, 1/params.dt_sim]));
    
               # update the row index to point towards the next msmt
               r_ind= r_ind + 1;
               if i == 0:
                   # plug steering angle and wheel speed model in A
                   [fart,S,V]= np.linalg.svd( estimator.D_bar );
                   r_S= np.linalg.rank(S)
                   D_bar_p= np.dot(np.sqrtm( np.inv(S[0:r_S,0:r_S]) ),np.transpose(V[:, 0:r_S]));
        
                   obj.A[ r_ind : r_ind + r_S - 1, c_ind : c_ind + params.m - 1]= np.dot(D_bar_p,estimator.Phi_k);
        
                   obj.A[ r_ind : r_ind + r_S - 1, c_ind + params.m : c_ind + 2*params.m - 1]= -D_bar_p;
        
                   # update the row & column indexes
                   r_ind= r_ind + r_S;
                   c_ind= c_ind + params.m;
        
                   # plug lidar model in A
                   obj.A[  r_ind : r_ind + estimator.n_k - 1,c_ind : c_ind + params.m - 1 ]= np.dot(np.kron( np.eye( estimator.n_L_k ) , params.sqrt_inv_R_lidar ),estimator.H_k);
        
                   # record lidar msmt indieces in A
                   obj.abs_msmt_ind= obj.abs_msmt_ind
               else:
                   # plug steering angle and wheel speed model in A
                   [fart,S,V]= np.linalg.svd( obj.D_bar_ph[ i ] );
                   r_S= np.linalg.rank(S);
                   D_bar_ph_p= np.dot(np.sqrtm( np.inv(S[0:r_S,0:r_S]) ),np.transpose(V[:,0:r_S]));
        
                   obj.A[ r_ind : r_ind + r_S - 1, c_ind : c_ind + params.m - 1 ]= np.dot(D_bar_ph_p,obj.Phi_ph[ i + 1 ]);
        
                   obj.A[ r_ind : r_ind + r_S - 1, c_ind + params.m : c_ind + 2*params.m -1]= -D_bar_ph_p;
        
                   # update the row & column indexes
                   r_ind= r_ind + r_S;
                   c_ind= c_ind + params.m;
        
                   # lidar Jacobian part
                   n_L_i= np.dot(obj.n_ph( i ),np.inv( params.m_F));
                   obj.A[ r_ind : r_ind + obj.n_ph[ i ] - 1,c_ind : c_ind + params.m - 1]= np.kron( np.eye(n_L_i) , np.dot(params.sqrt_inv_R_lidar ),obj.H_ph[i-1]));
        
                   # record lidar msmt indieces in A
                   obj.abs_msmt_ind=obj.abs_msmt_ind
        
                   # update the row index
                   r_ind= r_ind + obj.n_ph[i];
