import numpy as np
from scipy.linalg import expm
import sys
sys.path.insert(0,'../functions/')
import pi_to_pi
import FG_fn
import R_NB_rot
import Q_BE_fn
import body2nav_3D
import math


class EstimatorClassSlam:
     
      landmark_map = None

      XX= np.zeros((15,1))
      x_true= np.zeros((3,1))
      alpha =  None # state of interest extraction vector
      PX= np.zeros((15,15))
                
      association =  None # association of current features
      association_full =  None # association of current features
      association_true =  None # only for simulation
      association_no_zeros =  None # association of associated features
      num_landmarks= 0  # nunber of landmarks in the map
      num_associated_lms= 0
      num_extracted_features  =  None
      num_of_extracted_features  =  None
      number_of_associated_LMs  =  None    
      n_k =  None # number of absolute measurements at current time
      num_faults_k =  None # number of injected faults at current time
      gamma_k  =  None
      q_k  =  None
      Y_k  =  None
      H_k  =  None
      L_k  =  None
      Phi_k   =  None # state evolution matrix
      D_bar =  None # covariance increase for the state evolution    
      T_d= 0  # detector threshold
      q_d= 0  # detector for the window of time
      initial_attitude =  None # save initial attitude for the calibration of IM?U biases
      appearances= np.zeros(300) # if there are more than 300 landmarks, something's wrong
      FoV_landmarks_at_k =  None # landmarks in the field of view
      current_wp_ind= 1  # index of the sought way point
      goal_is_reached= 0
      steering_angle= 0
      lm_ind_fov =  None # indexes of the landmarks in the field of view       
      M= 0 # preceding horizon size in epochs
      x_ph =  None # poses in the time window
      z_fg =  None # all the msmts in the time window
      z_lidar_ph =  None # lidar msmts in the ph
      z_lidar =  None # current lidar msmts
      z_gyro= 0  # current gyro msmt
      z_gyro_ph =  None # gyro msmts in the ph
      PX_prior =  None # cov matrix of the prior
      Gamma_prior =  None # information matrix of the prior
      m_M =  None # number of states to estimate
      n_total =  None # total numbe of msmts
      association_ph =  None # associations during the ph
      odometry_k =  None # odometry msmts at the current time
      odometry_ph =  None # velocity and steering angle for the ph
      x_prior =  None # stores x_{k-M} as a msmt for the next epoch
      n_L_k= 0 # number of associations at k
      n_L_M= 0 # number of associations in the ph
      H_k_gps =  None
      H_k_lidar =  None
      n_gps_k =  None
      n_L_k_ph =  None # number of associations in the ph



      def __init__(self,imu_calibration_msmts, params):
            
          # Initial attitude
          self.initialize_pitch_and_roll(imu_calibration_msmts)
          # initialize the yaw angle
          self.XX[params.ind_yaw]= np.deg2rad(params.initial_yaw_angle)

          # save initial attitude for calibration
          self.initial_attitude= self.XX[6:9]

          # initialize covariance
          self.PX[9:12, 9:12]= np.diag( np.array([params.sig_ba,params.sig_ba,params.sig_ba]) )**2
          self.PX[12:15, 12:15]= np.diag( np.array([params.sig_bw,params.sig_bw,params.sig_bw]) )**2
      # ----------------------------------------------
      # ----------------------------------------------                    
      def compute_alpha(self,params):
          self.alpha= np.array([[-math.sin( self.XX(params.ind_yaw) )]
                      [math.cos( self.XX(params.ind_yaw) )]
                       [0] ])
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
          for i in range(1,self.n_L_k):
              # Indexes
              indz= 2*i + np.array([-1,0]);
              dx= self.landmark_map[self.lm_ind_fov[i], 1] - self.XX[0];
              dy= self.landmark_map[self.lm_ind_fov[i], 2] - self.XX[1];
    
              # Jacobian -- H
              self.H_k_lidar[indz,0]= np.array([[-cpsi], [spsi]]);
              self.H_k_lidar[indz,1]= np.array([[-spsi], [-cpsi]]);
              self.H_k_lidar[indz,params.ind_yaw]= np.concatenate((np.dot(-dx,spsi) + np.dot(dy,cpsi),np.dot(-dx,cpsi) - np.dot(dy,spsi)),axis = 1)               

              # compute the whiten jacobian matrix for lidar msmts
              self.H_k_lidar= np.dot( np.kron( np.eye( self.n_L_k ) , params.sqrt_inv_R_lidar ), self.H_k_lidar );

      # ----------------------------------------------
      # ----------------------------------------------  
      def initialize_pitch_and_roll(self, imu_calibration_msmts):
          # calculates the initial pitch and roll
          # compute gravity from static IMU measurements
          g_bar= np.mean( imu_calibration_msmts,1 )
          # Books method
          g_bar= -g_bar
          self.XX[6]= math.atan2( g_bar[1],g_bar[2] )
          self.XX[7]= math.atan2( -g_bar[0], np.sqrt( g_bar[1]**2 + g_bar[2]**2 ) )
          # My method -- works for z-axis pointing down (accz < 0)
          # theta=  atan2( g_bar(1) , abs(g_bar(3)) )
          # phi=   -atan2( g_bar(2) , abs(g_bar(3)) )
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
          [F,G]= FG_fn.FG_fn(u[0],u[1],u[2],u[4],u[5],self.XX[6] ,self.XX[7],self.XX[8],self.XX[9],self.XX[10],self.XX[11],self.XX[13],self.XX[14],taua,tauw)

          # Discretize system for IMU time (only for variance calculations)
          self.discretize(F, G, S, dT);
          
      # ----------------------------------------------

      def calibration(self, imu_msmt, params):
          # create a fake msmt and do a KF update to calibrate biases.
          # Also increase D_bar so that biases keep changing (not too
          # small variance)

          # create a fake msmt and make a KF update
          z= np.array([[np.zeros((6,1))],[ self.initial_attitude]])

          # Calibration msmt update
          L= np.dot( np.dot( self.PX[0:15,0:15] * np.transpose(params.H_cal) ), np.linalg.inv( np.dot( np.dot( params.H_cal, self.PX[0:15,0:15] ), np.transpose(params.H_cal) ) + params.R_cal ) )
          z_hat= np.dot( params.H_cal, self.XX[0:15] )
          innov= z - z_hat
          innov= pi_to_pi.pi_to_pi(innov)
          self.XX[0:15]= self.XX[0:15] + np.dot( L, innov )
          self.PX[0:15,0:15]= self.PX[0:15,0:15] - np.dot( np.dot( L, params.H_cal ), self.PX[0:15,0:15] )

          # linearize and discretize after every non-IMU update
          tmp = self.linearize_discretize( imu_msmt, params.dt_imu, params)
          self.Phi_k = tmp[0]
          self.D_bar = tmp[1]
          # If GPS is calibrating initial biases, increse bias variance
          self.D_bar[9:12,9:12]= self.D_bar[9:12,9:12] + np.diag( [params.sig_ba**2,params.sig_ba**2,params.sig_ba**2] )
          self.D_bar[12:15,12:15]= self.D_bar[12:15,12:15] + np.diag( [params.sig_bw**2,params.sig_bw**2,params.sig_bw**2] )

      # ----------------------------------------------
      # ----------------------------------------------  

      def imu_update(self, imu_msmt, params ):
          # updates the state with the IMU reading, NO cov update
          # Create variables (for clarity)
          v= self.XX[3:6]
          phi= self.XX[6]
          theta= self.XX[7] 
          psi= self.XX[8]
          b_f= self.XX[9:12].transpose()
          b_w= self.XX[12:15].transpose()
          f= imu_msmt[0:3]
          w= imu_msmt[3:6]

          if (params.SWITCH_CALIBRATION ==1):
             taua= params.taua_calibration
             tauw= params.tauw_calibration
          else:
             taua= params.taua_normal_operation
             tauw= params.tauw_normal_operation

          # Calculate parameters
          R_NB= R_NB_rot.R_NB_rot(phi,theta,psi) #<============
          Q_BE= Q_BE_fn.Q_BE_fn(phi,theta)
          r_dot= v
          v_dot= np.dot( R_NB, (f - b_f).transpose() ) + params.g_N
          E_dot= np.dot( Q_BE, (w - b_w).transpose() )
          b_f_dot= np.dot( -np.eye(3) / taua, b_f.transpose())
          b_w_dot= np.dot( -np.eye(3) / tauw, b_w.transpose())
          x_dot= np.concatenate((r_dot,v_dot,E_dot,b_f_dot,b_w_dot),axis=0)
          # update estimate
          self.XX[0:15]= self.XX[0:15] + params.dt_imu * x_dot
          self.PX[0:15,0:15]= np.dot( np.dot(self.Phi_k, self.PX[0:15,0:15]), np.transpose(self.Phi_k) ) + self.D_bar
      # ----------------------------------------------
      # ----------------------------------------------
      def yaw_update(self,w, params):

          n_L= int((self.XX.shape[0] - 15) / 2)
          H= np.zeros((1, 15 + int(2*n_L)))
          H[0,8]= 1
          R= params.R_yaw_fn( np.linalg.norm(self.XX[3:6]));
          z= self.yawMeasurement(w,params)
          L= np.dot( np.dot( self.PX, np.transpose(H) ), np.linalg.inv(np.dot( np.dot( H, self.PX ), np.transpose(H) ) + R) )
          innov= z - np.dot(H, self.XX)
          innov= pi_to_pi.pi_to_pi(innov)
          self.XX[8]= pi_to_pi.pi_to_pi(self.XX[8])
          self.XX= self.XX + np.dot( L, innov )
          self.PX= self.PX - np.dot( np.dot( L, H ), self.PX )
      # ----------------------------------------------
      # ----------------------------------------------
      def yawMeasurement(self, w, params):
          r= np.array([[-params.r_IMU2rearAxis],[0],[0]])
          v_o= self.XX[3:6]
          R_NB= R_NB_rot.R_NB_rot( self.XX[6], self.XX[7], self.XX[8])
          v_a= v_o + np.dot( R_NB, (np.cross(w,r.transpose())).transpose() )
          v_a= v_a / np.linalg.norm(v_a)
          yaw= math.atan2(v_a[1],v_a[0]) 
          return yaw
      # ----------------------------------------------
      # ----------------------------------------------
      def vel_update_z(self, R):
          # Normalize yaw
          self.XX[8]= pi_to_pi.pi_to_pi( self.XX[8] )
          # Update
          R_BN= np.transpose(R_NB_rot.R_NB_rot( self.XX[6], self.XX[7], self.XX[8] ))
          H= np.dot( np.dot( np.array([0,0,1]), R_BN ), np.array([np.zeros(3),np.eye(3),np.zeros((3,9))]) )
          L= np.dot( np.dot( self.PX, np.transpose(H) ), np.linalg.inv( np.dot( np.dot( H, self.PX ), np.transpose(H) ) + R ) )
          z_hat= np.dot( H, self.XX )
          innov= 0 - z_hat

          self.XX= self.XX + np.dot( L, innov )
          self.PX= self.PX - np.dot( np.dot( L, H ), self.PX )

          # This is a different option to do the update in Z, but it is more
          # computationally expensive and does not offer better results in my case
          '''
          R_BN= np.transpose(R_NB_rot( x[6,k], x[7,k], x[8,k] ))
          H_virt= H_fn.H_fn(x[3,k], x[4,k], x[5,k], x[6,k], x[7,k], x[8,k])
          L= P*np.transpose(H_virt) / (H_virt*P*np.transpose(H_virt) + R_virt_Z)
          z= 0
          z_hat= np.transpose([0,0,1]) * R_BN * x[3:5,k+1]
          innov= z - z_hat
          x[:,k]= x[:,k] + L*innov
          P= P - L*H_virt*P
          '''       
      # ----------------------------------------------
      # ----------------------------------------------
      def gps_update(self, z, R, params):
          n_L= ((self.XX).shape[0] - 15) / 2
          # if we are fast enough --> use GPS velocity msmt
          if (np.linalg.norm(z[3:6]) > params.min_vel_gps and params.SWITCH_GPS_VEL_UPDATE==1): # sense velocity
             R= np.diag( R )
             H = np.concatenate((np.eye(6), np.zeros((6,9)), np.zeros((6,int(n_L*2)))),axis=1)
             print('GPS velocity')
    
           # update only the position, no velocity
          else:
             z= z[0:3]
             R= np.diag( R[0:3] )
             H= np.concatenate((np.concatenate((np.eye(3), np.zeros((3,12))),axis = 1), np.zeros((3,int(n_L*2)))),axis = 1)
             print('-------- no GPS velocity ---------')
          self.XX[8]= pi_to_pi.pi_to_pi( self.XX[8] )
          L= np.dot( np.dot(self.PX, np.transpose(H)), np.linalg.inv( np.dot( np.dot(H, self.PX), np.transpose(H) ) + R) )
          innov= z - (np.dot( H, self.XX )).transpose()
          self.XX= self.XX + np.dot(L, innov.transpose())
          self.PX= self.PX - np.dot( np.dot(L, H), self.PX)
      # ----------------------------------------------
      # ----------------------------------------------
      def lidar_update(self,z,association,params):

          
          R= params.R_lidar;
          self.XX[8]= pi_to_pi.pi_to_pi( self.XX[8] );

          if np.all(association == -1):
             return 0
          # Eliminate the non-associated features
          ind_to_eliminate= (association == -1) | (association == -2);

          tmp_list = []
          check= np.shape(ind_to_eliminate)
          notScalar= len(check)
          if (notScalar == 0):
            if (ind_to_eliminate == 1):
               z=[]
               association = []
               return 0
          else:
            for i in range(ind_to_eliminate.shape[0]):
                if (ind_to_eliminate[i] == 1):
                   tmp_list.append(i)
            z = np.delete(z,tmp_list,axis = 0)
            association = np.delete(association,tmp_list)

          # Eliminate features associated to landmarks that has appeared less than X times
          ind_to_eliminate = []
          for i in range(association.shape[0]):
              if (self.appearances[int(association[i])] <= params.min_appearances):
                 ind_to_eliminate.append(1)
              else:
                 ind_to_eliminate.append(0)
          
          ind_to_eliminate= np.array(ind_to_eliminate)

          tmp_list = []
          check= np.shape(ind_to_eliminate)
          notScalar= len(check)
          if (notScalar == 0):
            if (ind_to_eliminate == 1):
               z=[]
               association = []
               return 0
          else:
            for i in range(ind_to_eliminate.shape[0]):
                if (ind_to_eliminate[i] == 1):
                   tmp_list.append(i)
            z = np.delete(z,tmp_list,axis = 0)
            association = np.delete(association,tmp_list)
          

          # if no measurent can be associated --> return
          if not(z.any()):
              return 0

          lenz= association.shape[0];
          lenx= self.XX.shape[0];

          R= np.kron( R,np.eye(lenz) );
          H= np.zeros((2*lenz,lenx));

          #Build Jacobian H
          spsi= float( math.sin(self.XX[8]) );
          cpsi= float( math.cos(self.XX[8]) );
          zHat= np.zeros(int(2*lenz));
          for i in range(association.shape[0]):
              # Indexes
              indz= np.array([ int(2*i), int(2*i + 1) ]);
              indx= [ int(15 + 2*association[i]), int(15 + 2*association[i] + 1) ];
    
              dx= float(self.XX[indx[0]] - self.XX[0]);
              dy= float(self.XX[indx[1]] - self.XX[1]);

              # Predicted measurement
              zHat[[indz[0],indz[1]]]= np.array([dx*cpsi + dy*spsi,-dx*spsi + dy*cpsi]);
    
              # Jacobian -- H
              H[indz[0],0]= -cpsi
              H[indz[1],0]= spsi
              H[indz[0],1]= -spsi
              H[indz[1],1]= -cpsi
              H[indz[0],8]= -dx * spsi + dy * cpsi
              H[indz[1],8]= -dx * cpsi - dy * spsi
              H[indz[0],indx[0]]= cpsi
              H[indz[1],indx[0]]= -spsi
              H[indz[0],indx[1]]= spsi
              H[indz[1],indx[1]]= cpsi
          zHat= np.transpose([zHat])

          # Update
          Y= np.dot( np.dot( H, self.PX ), np.transpose(H) ) + R;
          L= np.dot( np.dot( self.PX, np.transpose(H) ), np.linalg.inv(Y) );
          z_size= np.shape(z)
          zVector= z.reshape([int(z_size[0]*z_size[1]),1]);
          innov= zVector - zHat;

          # If it is calibrating, update only landmarks
          if (params.SWITCH_CALIBRATION ==1):
              XX0= self.XX[0:15];
              PX0= self.PX[0:15,0:15];
              self.XX= self.XX + np.dot( L, innov );
              self.PX= self.PX - np.dot( np.dot( L, H ), self.PX );
              self.XX[0:15]= XX0;
              self.PX[0:15,0:15]= PX0;
          else:
              if (self.XX).shape[1] > 1:
                input('start')
              self.XX= self.XX + np.dot( L, innov);
              self.PX= self.PX - np.dot( np.dot( L, H ), self.PX );
      # ----------------------------------------------
      # ----------------------------------------------    
      def addNewLM(self, z, R):

          # Number of landmarks to add
          n_L= z.shape[0];

          # update total number of landmarks
          self.num_landmarks= self.num_landmarks + n_L;

          # Add new landmarks to state vector
          z= body2nav_3D.body2nav_3D(z,self.XX[0:9]);
          z_size= np.shape(z)
          zVector= np.reshape(z, [z_size[0]*z_size[1],1])
          self.XX = np.concatenate((self.XX,zVector),axis=0)

          spsi= math.sin(self.XX[8]);
          cpsi= math.cos(self.XX[8]);
          for i in range(n_L):
              ind= np.array([(15 + 2*i),(15 + 2*i + 1)])
              dx= float(self.XX[ind[0]] - self.XX[0]);
              dy= float(self.XX[ind[1]] - self.XX[1]);
    
              H= np.array([[-cpsi, -spsi, -dx*spsi + dy*cpsi],[spsi,  -cpsi, -dx*cpsi - dy*spsi]]) 
              PX_2d= np.array([[float(self.PX[0,0]), float(self.PX[0,1]), float(self.PX[0,8])],[float(self.PX[1,0]), float(self.PX[1,1]), float(self.PX[1,8])],[float(self.PX[8,0]), float(self.PX[8,1]), float(self.PX[8,8])]])
              Y= np.dot( np.dot(H, PX_2d), np.transpose(H)) + R
              
              Y_size = np.shape(Y)
              PX_size = np.shape(self.PX)
              self.PX = np.concatenate((self.PX,np.zeros([PX_size[1],Y_size[1]])),axis=1)
              tmp= np.concatenate((np.zeros([Y_size[0],PX_size[1]]),Y),axis=1)
              self.PX = np.concatenate((self.PX,tmp),axis=0)
      # ----------------------------------------------
      # ----------------------------------------------    
      def increase_landmarks_cov(self, minPXLM):

          if ((self.PX).shape[0] == 15): 
             return 0 
          PXLM= np.diag( self.PX[15:,15:] )
          newDiagLM=[]
          for i in range(PXLM.shape[0]):
            newDiagLM.append(max(PXLM[i],minPXLM))
          
          newDiagLM= np.array(newDiagLM);
          diffDiagLM= PXLM - newDiagLM;
          self.PX[15:,15:]= self.PX[15:,15:] - np.diag( diffDiagLM )

# =====================================================================================  
      def discretize(self,F, G, S, dT):
          #MATRICES2DISCRETE This def discretize the continuous time model. It
          #works for either the GPS or IMU discretization times.
          # sysc= ss(F, zeros(15,1), zeros(1,15), 0)
          # sysd= c2d(sysc, dT)
          # Phi= sysd.A
          # Methdo to obtain covariance matrix for dicrete system
          C= np.concatenate((np.concatenate((-F, np.dot(G,np.dot(S,np.transpose(G)))),axis = 1),np.concatenate((np.zeros([15,15]), np.transpose(F)),axis = 1)),axis = 0)

          # Proper method
          EXP= expm(C*dT)
          self.Phi_k= np.transpose(EXP[15:,15:])
          #self.D_bar= self.Phi_k * EXP[0:15,15:]

          # Simplified method
          self.D_bar= np.dot( np.dot( (G*dT) , (S/dT) ) , np.transpose((G*dT)) ) # simplified version
      # ----------------------------------------------
      # ----------------------------------------------
      def nearest_neighbor(self, z, params):

          n_F= z.shape[0];
          n_L= int((self.XX.shape[0] - 15) / 2);
          association= np.ones(n_F) * (-1);
          if (n_F == 0 or n_L == 0):
             return association
 
          spsi= math.sin(self.XX[8]);
          cpsi= math.cos(self.XX[8]);
          zHat= np.zeros(2);
          # Loop over extracted features
          for i in range(n_F):
              minY= params.threshold_new_landmark;
    
              for l in range(n_L):
                  ind= np.array([15 + 2*l,15 + 2*l +1])
        
                  dx= float(self.XX[ind[0]] - self.XX[0]);
                  dy= float(self.XX[ind[1]] - self.XX[1]);
        
                  zHat[0]=  dx*cpsi + dy*spsi;
                  zHat[1]= -dx*spsi + dy*cpsi;
                  gamma= z[i,:] - zHat;
        
                  H= np.array([[-cpsi, -spsi, -dx*spsi + dy*cpsi,  cpsi, spsi],
                      [spsi, -cpsi, -dx*cpsi - dy*spsi, -spsi, cpsi]]);
                  
                  PX_2d= np.array([[float(self.PX[0,0]), float(self.PX[0,1]), float(self.PX[0,8]), float(self.PX[0,ind[0]]), float(self.PX[0,ind[1]])],[float(self.PX[1,0]), float(self.PX[1,1]), float(self.PX[1,8]), float(self.PX[1,ind[0]]), float(self.PX[1,ind[1]])],[float(self.PX[8,0]), float(self.PX[8,1]), float(self.PX[8,8]), float(self.PX[8,ind[0]]), float(self.PX[8,ind[1]])],[float(self.PX[ind[0],0]), float(self.PX[ind[0],1]), float(self.PX[ind[0],8]), float(self.PX[ind[0],ind[0]]), float(self.PX[ind[0],ind[1]])],[float(self.PX[ind[1],0]), float(self.PX[ind[1],1]), float(self.PX[ind[1],8]), float(self.PX[ind[1],ind[0]]), float(self.PX[ind[1],ind[1]])]])

                  Y= np.dot(np.dot(H,PX_2d),np.transpose(H)) + params.R_lidar;
                  y2= np.dot(np.dot(np.transpose(gamma),np.linalg.inv(Y)),gamma);
                  if (y2 < minY):
                      minY= y2;
                      association[i]= l;

              # If the minimum value is very large --> new landmark
              if (minY > params.T_NN and minY < params.threshold_new_landmark):
                  association[i]= -2;


          # Increase appearances counter
          for i in range(n_F):
              if (association[i] != -1 and association[i] != -2):
                  self.appearances[int(association[i])]= self.appearances[int(association[i])] + 1;
          return association


