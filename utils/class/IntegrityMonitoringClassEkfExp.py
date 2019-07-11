import numpy as np
import math
from scipy.stats import norm
from scipy.stats.distributions import chi2
from scipy.stats import ncx2
import scipy.optimize as sciopt
from scipy.special import comb
from scipy.sparse.linalg import eigs
from scipy.stats import chi2


class IntegrityMonitoringClassEkfExp:
    m = 3
    calculate_A_M_recursively = 0
    C_req = None
    p_hmi = None
    detector_threshold = None
    is_extra_epoch_needed = -1  # initialize as (-1), then a boolean
    ind_im = np.array([0, 1, 8])

    # (maybe unnecessary)
    E = None
    B_bar = None

    # hypotheses
    inds_H = None  # faulted indexes under H hypotheses
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
    M = 0  # size of the preceding horizon in epochs
    n_M = None  # num msmts in the preceding horizon (including k) -if FG ---> num abs msmts
    n_L_M = None  # num landmarks in the preceding horizon (including k)
    Phi_M = None
    q_M = None
    gamma_M = None
    Y_M = []
    A_M = np.array([])
    M_M = None
    P_MA_M = None
    P_F_M = None

    # preceding horizon saved (ph)
    Phi_ph = []
    q_ph = []
    gamma_ph = []
    A_ph = None
    L_ph = []
    Lpp_ph = []
    H_ph = []
    Y_ph = []
    P_MA_ph = []
    n_ph = []
    n_F_ph = None  # number of features associated in the preceding horizon

    # Factor Graph variables
    m_M = None      # number of states to estimate
    n_total = None   # total number of msmts (prior + relative + abs)
    XX_ph = None
    D_bar_ph = None
    A = None
    Gamma_fg = None  # information matrix
    M_fg = None
    PX_prior = None
    PX_M = None
    abs_msmt_ind = None
    faulted_LMs_indices = None
    Gamma_prior = None
    lidar_msmt_ind = None
    gps_msmt_ind = None
    n_gps_ph = None  # number of gps msmt at each epoch in PH
    H_gps_ph = None
    H_lidar_ph = None
    n_M_gps = None
    A_reduced = None
    min_f_dir_vs_M_dir = None
    f_mag = None
    noncentral_dof = [None]*10000  # cell(10000,1)
    f_dir_sig2 = [None]*10000  # cell(10000,1)
    M_dir = [None]*10000  # cell(10000,1)
    counter_H = 0
    # ======================================================================================================

    def __init__(self, params, estimator):
        # if the preceding horizon is fixed in epochs --> set M
        if (params.SWITCH_FIXED_LM_SIZE_PH == 1):
            self.M = 0
        else:
            self.M = params.M

        # continuity requirement
        self.C_req = params.continuity_requirement

        # initialize the preceding horizon
        #self.n_ph = np.zeros((params.M, 1))
        #self.Phi_ph = [None]*(params.M + 1)  # np.cell( 1, params.M + 1 ); # need an extra epoch here
        #self.H_ph = [None]*(params.M)  # np.cell( 1, params.M );
        #self.gamma_ph = [None]*(params.M)  # np.cell(1, params.M);
        #self.q_ph = np.ones((params.M, 1)) * (-1)
        #self.L_ph = [None]*(params.M)  # np.cell(1, params.M);
        #self.Lpp_ph = [None]*(params.M + 1)  # np.cell(1, params.M + 1); # need an extra epoch here (osama)
        #self.Y_ph = [None]*(params.M)  # np.cell(1, params.M);
        #self.P_MA_ph = [None]*(params.M)  # np.cell(1, params.M);
        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------

    def optimization_fn(self, f_M_mag, fx_hat_dir, M_dir, sigma_hat, l, dof):
        #print(self.T_d)
        #print(dof)
        #print(float(norm.cdf(l , f_M_mag * fx_hat_dir, sigma_hat)))
        #input(format(float(chi2.cdf(self.T_d, dof, f_M_mag**2 * M_dir )),'.2g'))
        neg_p_hmi = - ( (1 - float(norm.cdf(l , f_M_mag * fx_hat_dir, sigma_hat)) + float(norm.cdf(-l , f_M_mag * fx_hat_dir, sigma_hat))) * float(ncx2.cdf(self.T_d, dof, f_M_mag**2 * float(M_dir) )) )
        return float(neg_p_hmi)
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    def compute_E_matrix(self, i, m_F):
        if (np.sum(i) == 0):  # E matrix for only previous state faults
            self.E = np.zeros((self.m, self.n_M + self.m))
            self.E[:, (self.E.shape[1]-self.m):] = np.eye(self.m)
            print("I am in 0")
        else:  # E matrix with faults in the PH
            print("I am in 1")
            self.E = np.zeros((self.m + m_F*(i.shape[0]) , self.n_M + self.m))
            self.E[(self.E.shape[0]-self.m): , (self.E.shape[1]-self.m):]= np.eye(self.m) # previous bias
            for j in range(i.shape[0]):
                self.E[m_F*j : m_F*(j+1) , (int(i[j]))*m_F : (int(i[j])+1)*m_F ]= np.eye(m_F); # landmark i(j) faulted

    def monitor_integrity(self, estimator, counters, data, params):

        # keep only the elements for the [x-y-theta]
        # the state evolution matrix from one lidar msmt to the next one
        self.Phi_k = estimator.Phi_k  ######## CAREFUL
        for i in range(11):
           self.Phi_k= np.dot(self.Phi_k, estimator.Phi_k)
        
        Phi_k_2d= np.array([[float(self.Phi_k[0,0]), float(self.Phi_k[0,1]), float(self.Phi_k[0,8])],[float(self.Phi_k[1,0]), float(self.Phi_k[1,1]), float(self.Phi_k[1,8])],[float(self.Phi_k[8,0]), float(self.Phi_k[8,1]), float(self.Phi_k[8,8])]])

        self.Phi_k = Phi_k_2d

        # build L and H for the current time using only the pose indexes
        if (estimator.n_k == 0):  # no landmarks in the FoV at epoch k
            self.H_k = []
            self.L_k = []
        else:  # extract the indexes from the pose
            self.H_k = estimator.H_k[:, params.ind_pose]
            self.L_k = estimator.L_k[params.ind_pose-1, :]

        # current horizon measurements
        self.n_M= estimator.n_k + np.sum( np.array( self.n_ph ) )
        self.n_L_M = int( self.n_M/params.m_F )

        # the first time we have enough preceding horizon
        if (self.is_extra_epoch_needed == -1 and self.n_L_M >= params.min_n_L_M and counters.k_im > 1):
            self.is_extra_epoch_needed = 1
        # monitor integrity if the number of LMs in the preceding horizon is more than threshold
        if(( (params.SWITCH_FIXED_LM_SIZE_PH == 1) and (self.n_L_M >= params.min_n_L_M) and (self.is_extra_epoch_needed == False) ) or( (params.SWITCH_FIXED_LM_SIZE_PH==0) and (counters.k_im > self.M + 2) )):

            # Modify preceding horizon to have enough landmarks
            if (params.SWITCH_FIXED_LM_SIZE_PH == 1):
                self.n_M = estimator.n_k
                for i in range(len(self.n_ph)):
                    self.n_M = self.n_M + int(self.n_ph[i])
                    # if the preceding horizon is long enough --> stop
                    if (self.n_M >= params.min_n_L_M * params.m_F):
                        break
                # set the variables
                self.n_L_M = int(self.n_M/params.m_F)
                self.M = i + 1
            # common parameters
            alpha = np.array([[-math.sin(estimator.XX[params.ind_yaw])],[math.cos(estimator.XX[params.ind_yaw])], [0]]);

            PX_2d= np.array([[float(estimator.PX[0,0]), float(estimator.PX[0,1]), float(estimator.PX[0,8])],[float(estimator.PX[1,0]), float(estimator.PX[1,1]), float(estimator.PX[1,8])],[float(estimator.PX[8,0]), float(estimator.PX[8,1]), float(estimator.PX[8,8])]])
            self.sigma_hat = np.sqrt( np.dot(np.dot(np.transpose(alpha),PX_2d),alpha) );

            # detector threshold
            self.T_d = np.sqrt( chi2.ppf( 1 - float(params.continuity_requirement) , self.n_M ) );

            # If there are no landmarks in the FoV at k
            if (estimator.n_k == 0):
                self.Lpp_k = self.Phi_ph[0];
            else:
                self.Lpp_k = self.Phi_ph[0] - np.dot(np.dot(self.L_k, self.H_k), self.Phi_ph[0]);
            # accounting for the case where there are no landmarks in the FoV at
            # epoch k and the whole preceding horizon
            if (self.n_M == 0):
                self.Y_M = []
                self.A_M = []
                self.B_bar = []
                self.q_M = 0
                self.detector_threshold = 0
                self.p_hmi = 1
            else:
                # Update the innovation vector covarience matrix for the new PH
                self.compute_Y_M_matrix(estimator)

                # compute the A matrix for the preceding horizon
                self.compute_A_M_matrix(estimator)

                # compute B_bar matrix
                self.compute_B_bar_matrix(estimator)

                # M matrix
                self.M_M = np.dot(np.dot(np.transpose(self.B_bar), np.linalg.inv(self.Y_M)), self.B_bar)

                # set the threshold from the continuity req
                self.detector_threshold = chi2.ppf(1 - float(self.C_req), self.n_M)

                # compute detector
                self.q_M = np.sum(self.q_ph[0:self.M]) + estimator.q_k

                # TODO: very inefficient --> do not transform from cell to matrix
                self.P_MA_M = np.concatenate(([self.P_MA_k], np.array(self.P_MA_ph[0:self.M])), axis =1)[0]
                # fault probability of each association in the preceding horizon
                self.P_F_M = self.P_MA_M + float(params.P_UA);

                # compute the hypotheses (n_H, n_max, inds_H)
                self.compute_hypotheses(params)

                # initialization of p_hmi
                self.p_hmi= 0
                if (self.n_L_M - self.n_max < 2):  # need at least 5 msmts (3 landmarks) to monitor one landmark fault
                    print('Not enough redundancy: n_L_M = #d, n_max = #d\n', self.n_L_M, self.n_max)
                    self.p_hmi = 1;

                else:  # if we don't have enough landmarks --> P(HMI)= 1
                    self.P_H = np.ones((self.n_H, 1)) * math.inf  # initializing P_H vector
                    for i in range(0,self.n_H):
                        # build extraction matrix
                        if (i == 0):
                            self.compute_E_matrix(np.array([0]), params.m_F);
                        else:
                            self.compute_E_matrix(np.array([self.inds_H[i]]), params.m_F);
                        f_M_dir = np.dot(np.dot(np.dot(np.dot(np.transpose(self.E),np.linalg.inv(np.dot(np.dot(self.E, self.M_M), np.transpose(self.E)))),self.E), np.transpose(self.A_M)), alpha);
                        f_M_dir = f_M_dir/np.linalg.norm(f_M_dir); # normalize

                        # worst-case fault magnitude
                        fx_hat_dir= np.dot(np.dot(np.transpose(alpha),self.A_M),f_M_dir);
                        M_dir= np.dot(np.dot(np.transpose(f_M_dir),self.M_M),f_M_dir);

                        # worst-case fault magnitude
                        f_mag_min = 0
                        f_mag_max = 5
                        f_mag_inc = 5
                        p_hmi_H_prev = -1
                        opt_func= lambda f_M_mag, fx_hat_dir, M_dir, sigma_hat, l, dof: self.optimization_fn( f_M_mag, fx_hat_dir, M_dir, sigma_hat, l, dof )
                        bound = np.c_[f_mag_min, f_mag_max]
                        for k in range(10):
                            f_M_mag = sciopt.fminbound(opt_func, f_mag_min, f_mag_max, args=(fx_hat_dir, M_dir, self.sigma_hat, params.alert_limit, params.m_F*self.n_L_M))
                            p_hmi_H = self.optimization_fn( f_M_mag, fx_hat_dir, M_dir, self.sigma_hat, params.alert_limit, params.m_F*self.n_L_M )
                            # make it a positive number
                            p_hmi_H = -p_hmi_H
                            # check if the new P(HMI|H) is smaller
                            if (k == 0 or p_hmi_H_prev < p_hmi_H):
                                p_hmi_H_prev = p_hmi_H
                                f_mag_min = f_mag_min + f_mag_inc
                                f_mag_max = f_mag_max + f_mag_inc
                            else:
                                p_hmi_H = p_hmi_H_prev
                        # Add P(HMI | H) to the integrity risk
                        if (i == 0):
                            self.P_H_0 = np.prod(1 - self.P_F_M)
                            self.p_hmi = self.p_hmi + np.dot(p_hmi_H, self.P_H_0)
                        else:
                            # unfaulted_inds= all( 1:self.n_L_M ~= fault_inds(i,:)', 1 );
                            self.P_H[i] = np.prod(self.P_F_M[int(self.inds_H[i])]); #...
                            # * prod( 1 - P_F_M(unfaulted_inds)  );
                            self.p_hmi = self.p_hmi + p_hmi_H * self.P_H[i]
            input(self.p_hmi)
            # store integrity related data
            data.store_integrity_data(self, estimator, counters, params)
        # hey
        elif (counters.k_im > 0):  # if it's the first time --> cannot compute Lpp_k
            if (estimator.n_k == 0):
                self.Lpp_k = self.Phi_ph[0]
            else:
                self.Lpp_k = self.Phi_ph[0] - np.dot(np.dot(self.L_k, self.H_k), self.Phi_ph[0])

            if (params.SWITCH_FIXED_LM_SIZE_PH == 1):
                self.M = self.M + 1
                if (self.is_extra_epoch_needed == 1):
                    self.is_extra_epoch_needed = 0
        else:  # first time we get lidar msmts
            self.Lpp_k = 0
            if (params.SWITCH_FIXED_LM_SIZE_PH == 1):
                if (self.is_extra_epoch_needed == 1):
                    self.is_extra_epoch_needed = 0

        # store time
        data.im.time[counters.k_im] = counters.time_sim

        # update the preceding horizon
        self.update_preceding_horizon(estimator, params)

    def compute_A_M_matrix(self, estimator):
        # build matrix A_M in the first time
        if (self.A_M.any() == False or self.calculate_A_M_recursively == 0):
            
            # allocate L_k and initialize
            self.A_M = np.zeros((self.m, self.n_M + self.m))
            self.A_M[:, 0:estimator.n_k] = self.L_k

            for i in range(self.M):
                if (i == 0):
                    Dummy_Variable = self.Lpp_k
                else:
                    Dummy_Variable= np.dot(Dummy_Variable,self.Lpp_ph[i-1]);
                # if no landmarks in the FoV at some time in the preceding horizon
                if (self.n_ph[i] > 0):
                    n_start= int( estimator.n_k + np.sum( self.n_ph[0:i] ) );
                    n_end=   int( estimator.n_k + np.sum( self.n_ph[0:i+1] ) );
                    self.A_M[:,n_start : n_end]= np.dot(Dummy_Variable, self.L_ph[i]);

            # last entry with only PSIs
            self.A_M[:, self.n_M : self.n_M + self.m] = np.dot(Dummy_Variable,self.Lpp_ph[self.M-1])
        # calculate matrix A_M recusively
        else:
            self.A_M = np.array([self.L_k, self.Lpp_k*self.A_M])
            
            np.delete(self.A_M,np.arange(self.n_M,self.A_M.shape[1]-self.m),axis=1)
            self.A_M[:, self.A_M.shape[1]-self.m :] = np.dot(self.A_M[:, self.A_M.shape[1]-self.m :] ,np.linalg.inv(self.Lpp_ph[self.M]))

    def compute_B_bar_matrix(self, estimator):
        # Augmented B
        self.B_bar = math.inf*np.ones((self.n_M, self.n_M + self.m))
        A_prev = np.dot(np.linalg.inv(self.Lpp_k), self.A_M[ : , estimator.n_k : ])
        B_ind_row_start = estimator.n_k+1
        B_ind_col_end = estimator.n_k


        # accounting for the case where there are no landmarks in the FoV at epoch k
        if (estimator.n_k > 0):
            self.B_bar[0:estimator.n_k, :]  = np.concatenate(( np.eye(estimator.n_k), np.dot(np.dot(-self.H_k,self.Phi_ph[0]), A_prev) ), axis = 1)
            print(self.H_k.shape)
            print(self.Phi_ph[0].shape)
            input(A_prev.shape)

        # Recursive computation of B
        for i in range(self.M):
            A_prev = np.dot(np.linalg.inv(self.Lpp_ph[i]), A_prev[:, self.n_ph[i]:])
        # accounting for the case where there are no landmarks in the FoV at
        # one of the epochs in the preceding horizon
        if (self.n_ph[i] > 0):
            B = np.concatenate((np.eye(self.n_ph[i]), np.dot(np.dot(-self.H_ph[i], self.Phi_ph[i+1]),A_prev)),axis = 1);
            B_ind_row_end = B_ind_row_start + self.n_ph[i]
            self.B_bar[B_ind_row_start-1:B_ind_row_end, 0:B_ind_col_end] = 0
            self.B_bar[B_ind_row_start-1:B_ind_row_end, B_ind_col_end:] = B

            # increase row index for next element B
            B_ind_row_start = B_ind_row_start + self.n_ph[i]
            B_ind_col_end = B_ind_col_end + self.n_ph[i]

    def compute_hypotheses(self, params):
        # probability of "r" or more simultaneous faults
        flag_out = 0
        for r in range(self.P_F_M.shape[0]):
            if  ( (np.sum(self.P_F_M)**r) / math.factorial(r) ) < float(params.I_H):
                self.n_max = r-1
                flag_out = 1
                break

        # if no "r" holds --> all landmarks failing simultaneously must be monitored
        if (flag_out == 0):
            self.n_max = r

        if (self.n_max > 1):
            print('n_max:', self.n_max)
            if (params.SWITCH_ONLY_ONE_LM_FAULT == 1):
                self.n_max = 1

        # compute number of hypotheses
        self.n_H = 0
        self.inds_H = [None]*200
        start_ind = 0
        for num_faults in range(1,self.n_max+1):
            new_H= int(comb(self.n_L_M, num_faults));
            self.n_H= self.n_H + new_H;
            self.inds_H[ start_ind:start_ind+new_H ]= np.array(comb(np.arange(self.n_L_M), num_faults))
            start_ind= start_ind + new_H;
        del self.inds_H[start_ind:len(self.inds_H)]

    def compute_Y_M_matrix(self, estimator):
        # if it's the first epoch --> build the Y_M
        if len(self.Y_M) == 0:
            self.Y_M = np.zeros( (self.n_M, self.n_M) );

            self.Y_M[ 0:estimator.n_k, 0:estimator.n_k ]= estimator.Y_k;

            for i in range(self.M):
                n_start = estimator.n_k + np.sum( np.array( self.n_ph )[0:i] )
                n_end = estimator.n_k + np.sum( np.array( self.n_ph )[0:i+1] )
                self.Y_M[ n_start: n_end, n_start: n_end ]= self.Y_ph[i];
        else:  # update Y_M
            self.Y_M = np.concatenate( (np.concatenate( (estimator.Y_k, np.zeros((estimator.n_k,np.sum(np.array(self.n_ph[0:self.M]))))),axis = 1),np.concatenate((np.zeros((np.sum(np.array(self.n_ph[0:self.M])),estimator.n_k)), self.Y_M[0:np.sum(np.array(self.n_ph[0:self.M])), 0:np.sum(np.array(self.n_ph[0:self.M]))]),axis =1)) ,axis = 0 )

    def update_preceding_horizon(self, estimator, params):

            # TODO: organize
                if (params.SWITCH_FIXED_LM_SIZE_PH == 1):

                    self.n_ph.insert(0,estimator.n_k)
                    while len(self.n_ph) > self.M + 1:
                        self.n_ph.pop(len(self.n_ph)-1)

                    self.gamma_ph.insert(0,estimator.gamma_k)
                    while len(self.gamma_ph) > self.M + 1:
                        self.gamma_ph.pop(len(self.gamma_ph)-1)

                    self.q_ph.insert(0,estimator.q_k)
                    while len(self.q_ph) > self.M + 1:
                        self.q_ph.pop(len(self.q_ph)-1)

                    self.Phi_ph.insert(0,self.Phi_k)
                    while len(self.Phi_ph) > self.M + 2:
                        self.Phi_ph.pop(len(self.Phi_ph)-1)

                    self.H_ph.insert(0,self.H_k)
                    while len(self.H_ph) > self.M + 1:
                        self.H_ph.pop(len(self.H_ph)-1)

                    self.L_ph.insert(0,self.L_k)
                    while len(self.L_ph) > self.M + 1:
                        self.L_ph.pop(len(self.L_ph)-1)

                    self.Lpp_ph.insert(0,self.Lpp_k)
                    while len(self.Lpp_ph) > self.M + 1:
                        self.Lpp_ph.pop(len(self.Lpp_ph)-1)

                    self.Y_ph.insert(0,estimator.Y_k)
                    while len(self.Y_ph) > self.M + 1:
                        self.Y_ph.pop(len(self.Y_ph)-1)

                    self.P_MA_ph.insert(0,self.P_MA_k)
                    while len(self.P_MA_ph) > self.M + 1:
                        self.P_MA_ph.pop(len(self.P_MA_ph)-1)

                else:

                    self.n_ph.insert(0,estimator.n_k)

                    self.gamma_ph.insert(0,estimator.gamma_k)

                    self.q_ph.insert(0,estimator.q_k)

                    self.Phi_ph.insert(0,self.Phi_k)

                    self.H_ph.insert(0,self.H_k)

                    self.L_ph.insert(0,self.L_k)

                    self.Lpp_ph.insert(0,self.Lpp_k)

                    self.Y_ph.insert(0,estimator.Y_k)

                    self.P_MA_ph.insert(0,self.P_MA_k)
                #input(self.M)
                #print(self.n_ph)
                #print(self.gamma_ph)
                #print(self.q_ph)
                #print(self.Phi_ph)
                #print(self.H_ph)
                #print(self.L_ph)
                #print(self.Lpp_ph)
                #print(self.Y_ph)
                #input(self.P_MA_ph)
    def prob_of_MA(self, estimator, params):
          # dof of the non-central chi-square in the P(MA)
          chi_dof= self.m + params.m_F;
          # allocate memory
          spsi= float( math.sin(estimator.XX[params.ind_yaw]) );
          cpsi= float( math.cos(estimator.XX[params.ind_yaw]) );
          h_t= np.zeros((2,1));
          h_l= np.zeros((2,1));
          
          tmp = []
          for i in range(estimator.association.shape[0]):
              if estimator.association[i] == -1:
                tmp.append(i)

          estimator.association_no_zeros = np.delete(estimator.association,tmp,axis = 0)

          self.P_MA_k= np.ones(estimator.association_no_zeros.shape[0]) * (-1);
          self.P_MA_k_full= self.P_MA_k;

          # compute kappa
          if (self.A_M.any() == 0):
             self.mu_k = 0;
          elif (self.n_L_M - self.n_max < 2):
             self.kappa= 1;
             self.mu_k= np.dot(self.kappa,( np.sqrt(self.T_d) - np.sqrt( chi2.ppf(1 - params.I_MA , self.n_M) ) )**2);
          else:
              # compute Q matrix with A_M_(k-1) , Phi_(k-1), P_k, n_M_(k-1)
              PX_2d= np.array([[float(estimator.PX[0,0]), float(estimator.PX[0,1]), float(estimator.PX[0,8])],[float(estimator.PX[1,0]), float(estimator.PX[1,1]), float(estimator.PX[1,8])],[float(estimator.PX[8,0]), float(estimator.PX[8,1]), float(estimator.PX[8,8])]])
              Q= np.dot(np.dot(np.dot(np.dot(np.transpose(self.A_M), np.transpose(self.Phi_ph[0])), PX_2d), self.Phi_ph[0]), self.A_M);

              self.kappa= 0;
              C = comb(np.arange(self.n_L_M),self.n_max);#set of possible fault indices for n_max simultanous faults
              C = C+1
              for i in range(C.shape[0]):
                  # build extraction matrix
                  self.compute_E_matrix(np.array([C[i]]), params.m_F);
                  kappa_H= max(eigs( np.dot(np.dot(self.E,Q),np.transpose(self.E)))[0]) * min(eigs( np.dot(np.dot(self.E,self.M_M),np.transpose(self.E)))[0]);
                  # take the largest kappa
                  if (kappa_H > self.kappa):
                     self.kappa= kappa_H
              self.mu_k= np.dot(self.kappa ,( np.sqrt(self.T_d) - np.sqrt( chi2.ppf(1 - float(params.I_MA) , int(self.n_M)) ) )**2);

          #loop through each associated landmark
          for t in range (estimator.association_no_zeros.shape[0]):
              # take the landmark ID
              lm_id_t= int( estimator.association_no_zeros[t] );

              # initialize the P(MA)
              self.P_MA_k[t]= estimator.FoV_landmarks_at_k.shape[0] - 1;

              # build the necessary parameters
              landmark= estimator.landmark_map[ lm_id_t ];
              dx= float(landmark[0] - estimator.XX[0]);
              dy= float(landmark[1] - estimator.XX[1]);
              h_t[0]=  dx*cpsi + dy*spsi;
              h_t[1]= -dx*spsi + dy*cpsi;

              # loop through every possible landmark in the FoV (potential MA)
              for l in range(estimator.FoV_landmarks_at_k.shape[0]):
                  # take landmark ID
                  lm_id_l= int( estimator.FoV_landmarks_at_k[l] );

                  if (lm_id_t != lm_id_l):
                     # extract the landmark
                     landmark= estimator.landmark_map[lm_id_l ];

                     # compute necessary intermediate parameters
                     dx= float( landmark[0] - estimator.XX[0] );
                     dy= float( landmark[1] - estimator.XX[1] );
                     h_l[0]=  dx*cpsi + dy*spsi;
                     h_l[1]= -dx*spsi + dy*cpsi;
                     H_l= np.array([[-cpsi, -spsi, -dx*spsi + dy*cpsi],[spsi, -cpsi, -dx*cpsi - dy*spsi]])
                     y_l_t= h_l - h_t;
                     PX_2d= np.array([[float(estimator.PX[0,0]), float(estimator.PX[0,1]), float(estimator.PX[0,8])],[float(estimator.PX[1,0]), float(estimator.PX[1,1]), float(estimator.PX[1,8])],[float(estimator.PX[8,0]), float(estimator.PX[8,1]), float(estimator.PX[8,8])]])

                     Y_l= np.dot(np.dot(H_l,PX_2d),np.transpose(H_l)) + params.R_lidar;

                     # individual innovation norm between landmarks l and t
                     IIN_l_t= float( np.sqrt( np.dot(np.dot(np.transpose(y_l_t),np.linalg.inv(Y_l)),y_l_t) ) );

                     # if one of the landmarks is too close to ensure P(MA) --> set to one
                     if (IIN_l_t < np.sqrt(params.T_NN)):
                         self.P_MA_k[t]= 1;
                         break
                     else:
                         self.P_MA_k[t]= self.P_MA_k[t] - chi2.cdf( ( IIN_l_t - np.sqrt(params.T_NN) )**2 , chi_dof, self.mu_k );

              # store the P_MA for the full LMs
              # landmark selection
              if (params.SWITCH_LM_SELECTION==1):
                  if (self.P_MA_k[t] > params.P_MA_max):
                      self.P_MA_k[t]= -1;
                      estimator.association_no_zeros[t]= -1;

                      for i in range(estimator.association.shape[0]):
                          if estimator.association[i] == lm_id_t:
                            estimator.association[i] = -1

               # not more than probability one
              if (self.P_MA_k[t] > 1):
                  self.P_MA_k[t]= 1

          # remove non-associated ones
          if (params.SWITCH_LM_SELECTION == 1):

              tmp_list = []
              for i in range(self.P_MA_k.shape[0]):
                  if (self.P_MA_k[i] == -1):
                     tmp_list.append(i)
              self.P_MA_k = np.delete(self.P_MA_k,tmp_list,axis = 0)

              tmp_list = []
              for i in range(estimator.association_no_zeros.shape[0]):
                  if (estimator.association_no_zeros[i] == -1):
                     tmp_list.append(i)
              estimator.association_no_zeros = np.delete(estimator.association_no_zeros, tmp_list, axis = 0)
