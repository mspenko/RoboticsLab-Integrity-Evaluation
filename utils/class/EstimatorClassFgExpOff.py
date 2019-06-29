import numpy as np
import math
import scipy.io as sio

class EstimatorClassFgExpOff:
      EstimatorClassFgExpOff = None
      landmark_map = None

      XX= np.zeros((15,1))
      x_true= np.zeros((3,1))
      alpha = None # state of interest extraction vector
      PX= np.zeros(15)
       
      association = None # association of current features
      num_landmarks = None # number of landmarks in the map
        
      n_k = None # number of absolute measurements at current time
        
      q_k = None
      H_k = None
      Phi_k  = None  # state evolution matrix
      D_bar = None # covariance increase for the state evolution
        
      T_d= 0 # detector threshold
      q_d= 0 # detector for the window of time
        
      FoV_landmarks_at_k = None # landmarks in the field of view
      lm_ind_fov = None # indexes of the landmarks in the field of view
        
      M= 0 # preceding horizon size in epochs
      PX_prior = None # cov matrix of the prior
      Gamma_prior = None # information matrix of the prior
      m_M = None # number of states to estimate
      n_total = None # total numbe of msmts
      x_prior = None # stores x_{k-M} as a msmt for the next epoch
      n_L_k= 0 # number of associations at k
      n_L_M= 0 # number of associations in the ph
      H_k_gps = None
      H_k_lidar = None
      n_gps_k = None
      n_L_k_ph = None # number of associations in the ph

      def __init__(self,params):

          # initialize preceding horizon size
          if (params.SWITCH_FIXED_LM_SIZE_PH == 1):
              self.M= 0;
          else:
              self.M= params.M;
            
          # initialize to uninformative prior
          self.PX_prior= np.diag( np.dot(np.ones(params.m,1),eps) );
          # initialize covariance
          self.PX_prior[9:12, 9:12]= np.diag( [params.sig_ba,params.sig_ba,params.sig_ba] )**2;
          self.PX_prior[12:15,12:15]= np.diag( [params.sig_bw,params.sig_bw,params.sig_bw] )**2;
          self.Gamma_prior= np.inv(self.PX_prior);
          self.x_prior= np.zeros((params.m, 1));
          # allocate memory
          self.n_L_k_ph= np.zeros((params.M, 1));
         
          # load map if exists
          tdir = params.path+ 'landmark_map.mat'
          data = sio.loadmat(tdir)
          data = data['landmark_map']
          self.landmark_map= data;
          self.num_landmarks= self.landmark_map.shape[1]
      # ----------------------------------------------
      # ----------------------------------------------
      def compute_alpha(self,params):
          self.alpha= [[-math.sin( self.XX(params.ind_yaw) )],
                      [math.cos( self.XX(params.ind_yaw) )],
                      [np.zeros((13,1))]];
      # ----------------------------------------------
      # ----------------------------------------------    

      def linearize_discretize(self,u, dT, params):
          
          if (params.SWITCH_CALIBRATION==1):
             taua= params.taua_calibration;
             tauw= params.tauw_calibration;
             S= params.S_cal;
          else:
             taua= params.taua_normal_operation;
             tauw= params.tauw_normal_operation;
             S= params.S;

             # Compute the F and G matrices (linear continuous time)
             [F,G]=FG_fn.FG_fn(u[0],u[1],u[2],u[4],u[5],self.XX[6],self.XX[7],self.XX[8],self.XX[9],self.XX[10],self.XX[11],self.XX[13],self.XX[14],taua,tauw)
   
             # Discretize system for IMU time (only for variance calculations)
             self.discretize(F, G, S, dT);    
      # ----------------------------------------------
      # ---------------------------------------------- 
      def discretize(self,F, G, S, dT):
          #MATRICES2DISCRETE This def discretize the continuous time model. It
          #works for either the GPS or IMU discretization times.
          # sysc= ss(F, zeros(15,1), zeros(1,15), 0)
          # sysd= c2d(sysc, dT)
          # Phi= sysd.A
          # Methdo to obtain covariance matrix for dicrete system
          C= np.transpose(np.concatenate((np.concatenate((-F, np.dot(G,np.dot(S,np.transpose(G)))),axis = 0),np.concatenate((np.zeros([15,15]), np.transpose(F)),axis = 0)),axis = 1))
          # Proper method
          EXP= expm(C*dT)
          self.Phi_k= np.transpose(EXP[15:,15:])
          self.D_bar= self.Phi_k * EXP[0:15,15:]
          # Simplified method
          self.D_bar= np.dot( np.dot( (G*dT) , (S/dT) ) , np.transpose((G*dT)) ) # simplified version
      # ----------------------------------------------
      # ---------------------------------------------- 
      def compute_gps_H_k(self, params, FG, epoch):

          # check if there exist GPS msmt at the current lidar epoch
          if (isempty(FG.gps_R[epoch]) or (params.SWITCH_GPS_FG == 0)):
              self.n_gps_k= 0;
              self.H_k_gps= None;
          else:
              # lidar msmt noise cov matrix
              R= np.dot(np.concatenate((np.concatenate((params.mult_factor_pose_gps*eye(3),zeros(3)),axis= 0),np.concatenate((zeros(3),np.dot(params.mult_factor_vel_gps,eye(3))),axis = 0)),axis = 1),np.diag(FG.gps_R[epoch]))
              # compute the whiten jacobian matrix for GPS msmts
              self.H_k_gps= np.dot(np.sqrtm( np.inv(R) ),np.concatenate((np.eye(6), np.zeros((6,9))),axis = 0));
              # Number of GPS msmts
              self.n_gps_k= 6;
    
      # ----------------------------------------------
      # ---------------------------------------------- 
      def compute_imu_Phi_k( self, params, FG, epoch ):
          # linearize and discretize IMU model
          self.linearize_discretize(FG.imu[epoch], params.dt_imu, params)
    
          # process noise cov matrix beween two lidar epochs
          D_bar_init= self.D_bar;
          for i in range(1,12):
              self.D_bar= np.concatenate((self.Phi_k*self.D_bar*np.transpose(self.Phi_k)),axis=1) + D_bar_init;
          # imu jacobian matrix beween two lidar epochs
          self.Phi_k= self.Phi_k**12;
      # ----------------------------------------------
      # ---------------------------------------------- 
      def compute_lidar_H_k(self, params, FG, epoch): 
          # this funcion builds the Jacobian H of LMs msmts for the factor graphs...
          # case without actual mesaurements.

          spsi= math.sin(self.XX[9]);
          cpsi= math.cos(self.XX[9]);

          # landmarks in the field of view (saved from the online run)
          self.lm_ind_fov= FG.associations[epoch];

          # number of extracted landmarks (more specifically features)
          self.n_L_k= FG.associations[epoch].shape[0];

          # number of expected measurements
          self.n_k = np.dot(self.n_L_k,params.m_F);

          # build Jacobian
          self.H_k_lidar= np.zeros( (self.n_k , params.m) );
          for i in range(1:self.n_L_k):
              # Indexes
              indz= 2*i + [-1:0];
              dx= self.landmark_map[self.lm_ind_fov[i], 1] - self.XX[0];
              dy= self.landmark_map[self.lm_ind_fov[i], 2] - self.XX[1];
    
              # Jacobian -- H
              self.H_k_lidar[indz,0]= np.array([[-cpsi], [spsi]]);
              self.H_k_lidar[indz,1]= np.array([[-spsi], [-cpsi]]);
              self.H_k_lidar[indz,params.ind_yaw]= [-dx * spsi + dy * cpsi;
                                   -dx * cpsi - dy * spsi];
                          

              # compute the whiten jacobian matrix for lidar msmts
              self.H_k_lidar= kron( eye( self.n_L_k ) , params.sqrt_inv_R_lidar ) * self.H_k_lidar;

      # ----------------------------------------------
      # ---------------------------------------------- 
