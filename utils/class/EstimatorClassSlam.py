import numpy as np
import sys
sys.path.insert(0,'../functions/')
import pi_to_pi
import FG_fn
import R_NB_rot
import Q_BE_fn
import nearestNeighbor
import body2nav_3D


class EstimatorClassSlam:
     
      landmark_map = None

      XX= np.zeros((15,1))
      x_true= np.zeros((3,1))
      alpha =  None # state of interest extraction vector
      PX= np.zeros((15))
                
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
      appearances= np.zeros((1,300)) # if there are more than 300 landmarks, something's wrong
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
          self.initial_attitude= self.XX[7:9]

          # initialize covariance
          self.PX[9:1, 9:11]= np.diag( np.array([params.sig_ba,params.sig_ba,params.sig_ba]) )**2
          self.PX[12:14, 12:14]= np.diag( np.array([params.sig_bw,params.sig_bw,params.sig_bw]) )**2

          association= nearest_neighbor.nearest_neighbor(self, z, params)
      # ----------------------------------------------
      # ----------------------------------------------                    
      def compute_alpha(self,params):
          self.alpha= np.array([[-sin( self.XX(params.ind_yaw) )]
                      [cos( self.XX(params.ind_yaw) )]
                       [0] ])
      # ----------------------------------------------
      # ----------------------------------------------  
      def initialize_pitch_and_roll(self, imu_calibration_msmts):
          # calculates the initial pitch and roll
          # compute gravity from static IMU measurements
          g_bar= np.mean( imu_calibration_msmts,1 )

          # Books method
          g_bar= -g_bar
          self.XX[6]= np.atan2( g_bar[1],g_bar[2] )
          self.XX[7]= np.atan2( -g_bar[0], np.sqrt( g_bar[1]**2 + g_bar[2]**2 ) )

          # My method -- works for z-axis pointing down (accz < 0)
          # theta=  atan2( g_bar(1) , abs(g_bar(3)) )
          # phi=   -atan2( g_bar(2) , abs(g_bar(3)) )
      # ----------------------------------------------
      # ----------------------------------------------  

      def linearize_discretize(self,u,S,taua,tauw,dT):
          global XX
          # Compute the F and G matrices (linear continuous time)
          tmp= FG_fn.FG_fn(u[0],u[1],u[2],u[4],u[5],XX[6],XX[7],XX[8],XX[9],XX[10],XX[11],XX[13],XX[14],taua,tauw)
          F = tmp[0]
          G = tmp[1]
          # Discretize system for IMU time (only for variance calculations)
          tmp =discretize(F, G, S, dT)
          Phi = tmp[0]
          D_bar = tmp[1]

          return [Phi,D_bar]
      # ----------------------------------------------

      def calibration(self, imu_msmt, params):
          # create a fake msmt and do a KF update to calibrate biases.
          # Also increase D_bar so that biases keep changing (not too
          # small variance)

          # create a fake msmt and make a KF update
          z= np.array([[np.zeros((6,1))],[ self.initial_attitude]])

          # Calibration msmt update
          L= np.transpose(self.PX[0:14,0:14] * params.H_cal)/(params.H_cal*self.PX[0:14,0:14]*np.transpose(params.H_cal) + params.R_cal)
          z_hat= params.H_cal * self.XX[0:14]
          innov= z - z_hat
          innov= pi_to_pi.pi_to_pi(innov)
          self.XX[0:14]= self.XX[0:14] + L*innov
          self.PX[0:14,0:14]= self.PX[0:14,0:14] - L * params.H_cal * self.PX[0:14,0:14]

          # linearize and discretize after every non-IMU update
          tmp = self.linearize_discretize( imu_msmt, params.dt_imu, params)
          self.Phi = tmp[0]
          self.D_bar = tmp[1]
          # If GPS is calibrating initial biases, increse bias variance
          self.D_bar[9:11,9:11]= self.D_bar[9:11,9:11] + np.diag( [params.sig_ba,params.sig_ba,params.sig_ba] )**2
          self.D_bar[12:14,12:14]= self.D_bar[12:15,12:14] + np.diag( [params.sig_bw,params.sig_bw,params.sig_bw] )**2

      # ----------------------------------------------
      # ----------------------------------------------  

      def imu_update(self, imu_msmt, params ):
          # updates the state with the IMU reading, NO cov update
          # Create variables (for clarity)
          v= self.XX[3:5]
          phi= self.XX[6]
          theta= self.XX[7] 
          psi= self.XX[8]
          b_f= self.XX[9:11]
          b_w= self.XX[12:14]
          f= imu_msmt[0:2]
          w= imu_msmt[3:5]

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
          v_dot= R_NB * ( f - b_f ) + params.g_N
          E_dot= Q_BE * ( w - b_w )
          b_f_dot= -np.eye(3) / taua * b_f
          b_w_dot= -np.eye(3) / tauw * b_w
          x_dot= np.array([r_dot,v_dot,E_dot,b_f_dot,b_w_dot])

          # udpate estimate
          self.XX[0:14]= self.XX[0:14] + params.dt_imu * x_dot
          self.PX[0:14,0:14]= self.Phi_k * self.PX[0:14,0:14] * np.transpose(self.Phi_k) + self.D_bar
      # ----------------------------------------------
      # ----------------------------------------------
      def yawUpdate(self,w,R,r_IMU2rearAxis):

          global XX,PX

          n_L= (XX.shape[0] - 15) / 2
          H= np.zeros((1, 15 + 2*n_L))
          H[8]= 1

          z= yawMeasurement(w,r_IMU2rearAxis)
          L= PX*np.transpose(H) / (H*PX*np.transpose(H) + R)
          innov= z - H*XX
          innov= pi_to_pi.pi_to_pi(innov)
          XX[8]= pi_to_pi.pi_to_pi(XX[8])
          XX= XX + L*innov
          PX= PX - L*H*PX
      # ----------------------------------------------
      # ----------------------------------------------
      def yawMeasurement(self, w, params):
          r= np.array([[-params.r_IMU2rearAxis],[0],[0]])
          v_o= self.XX[3:5]
          R_NB= R_NB_rot.R_NB_rot( self.XX[6], self.XX[7], self.XX[8])
          v_a= v_o + R_NB * np.cross(w,r)
          v_a= v_a / np.norm(v_a)
          yaw= math.atan2(v_a[1],v_a[0]) 
          return yaw
      # ----------------------------------------------
      # ----------------------------------------------
      def vel_update_z(self, R):
          # Normalize yaw
          self.XX[8]= pi_to_pi.pi_to_pi( self.XX[8] )
          # Update
          R_BN= np.transpose(R_NB_rot.R_NB_rot( self.XX[6], self.XX[7], self.XX[8] ))
          H= np.array([0,0,1]) * R_BN * np.array([np.zeros(3),np.eye(3),np.zeros((3,9))])
          L= bj.PX*np.transpose(H) / (H*self.PX*np.transpose(H) + R)
          z_hat= H * self.XX
          innov= 0 - z_hat

          self.XX= self.XX + L*innov
          self.PX= self.PX - L*H*self.PX

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
      def vel_update_z(self, R):

          # Normalize yaw
          self.XX[8]= pi_to_pi.pi_to_pi( self.XX[8] )

          # Update
          R_BN= np.transpose(R_NB_rot.R_NB_rot( self.XX[6], self.XX[7], self.XX[8] ))
          H= np.array([0,0,1]) * R_BN * np.array([np.zeros(2),np.eye(2),np.zeros((2,8))])
          L= self.PX*np.transpose(H) / (H*self.PX*np.transpose(H) + R)
          z_hat= H * self.XX
          innov= 0 - z_hat

          self.XX= self.XX + L*innov
          self.PX= self.PX - L*H*self.PX
      # ----------------------------------------------
      # ----------------------------------------------

      def gps_update(self, z, R, params):
          n_L= ((self.XX).shape[0] - 15) / 2
          # if we are fast enough --> use GPS velocity msmt
          if (np.norm(z[3:4]) > params.min_vel_gps & params.SWITCH_GPS_VEL_UPDATE==1): # sense velocity
             R= np.diag( R )
             H= np.array([np.eye(6), np.zeros((6,9)), np.zeros(6,n_L*2)])
             print('GPS velocity')
    
           # update only the position, no velocity
          else:
             z= z[0:2]
             R= np.diag[ R[0:2] ]
             H= np.array([np.eye(3), np.zeros((3,12)), np.zeros((3,n_L*2))])
             print('-------- no GPS velocity ---------')

          self.XX[8]= pi_to_pi.pi_to_pi( self.XX[8] )
          L= self.PX*np.transpose(H) / (H*self.PX*np.transpose(H) + R)
          innov= z - H*self.XX
          self.XX= self.XX + L*innov
          self.PX= self.PX - L*H*self.PX
      # ----------------------------------------------
      # ----------------------------------------------
      def lidarUpdate(self,z,association,appearances,R,SWITCH_CALIBRATION):

          global XX,PX
          XX[8]= pi_to_pi.pi_to_pi(XX[8])
          if all(association == -1):
             return
          # Eliminate the non-associated features
          acc = 0
          for i in association:
              if(i == -1 or i ==0):
                 np.delete(z,acc,1)
              acc = acc+1

          acc = 0
          for i in association:
              if(i == -1 or i ==0):
                 np.delete(association,acc)
              acc = acc+1

          lenz= association.shape[0]
          lenx= XX.shape[0]

          R= np.kron((R,eye(lenz)))
          H= np.zeros((2*lenz,lenx))

          #Build Jacobian H
          spsi= math.sin(XX[8])
          cpsi= math.cos(XX[8])
          zHat= np.zeros((2*lenz,1))
          for i in range(association.shape[0]):
              # Indexes 
              indz= 2*i + np.array([-1,0])
              indx= 15 + 2*association[i]+ np.array([-1,0])

              dx= XX[indx[0]] - XX[0]
              dy= XX[indx[1]] - XX[1]

              # Predicted measurement
              zHat[indz]= np.array([[dx*cpsi + dy*spsi],[-dx*spsi + dy*cpsi]])

              # Jacobian -- H
              H[indz,0]= np.array([[-cpsi], [spsi]])
              H[indz,1]= np.array([[-spsi],[-cpsi]])
              H[indz,9]= np.array([[-dx * spsi + dy * cpsi],[-dx * cpsi - dy * spsi]])
              H[indz,indx]= np.array([[cpsi, spsi],[-spsi, cpsi]])

          # Update
          Y= H*PX*np.transpose(H) + R
          L= PX * np.transpose(H) / Y
          zVector= np.transpose(z)
          zVector= zVector[:]
          innov= zVector - zHat

          # If it is calibrating, update only landmarks
          if (SWITCH_CALIBRATION == 1):
             XX0= XX[0:14]
             PX0= PX[0:14,0:14]
             XX= XX + L*innov
             PX= PX - L*H*PX
             XX[0:14]= XX0
             PX[0:14,0:14]= PX0
          else: 
             XX= XX + L*innov
             PX= PX - L*H*PX
      # ----------------------------------------------
      # ----------------------------------------------    
      def addNewLM(self, z, R):

          # Number of landmarks to add
          n_L= z.shape[1];

          # update total number of landmarks
          self.num_landmarks= self.num_landmarks + n_L;

          # Add new landmarks to state vector
          z= body2nav_3D.body2nav_3D(z,self.XX[0:8]);
          zVector= np.transpose(z)
          zVector= zVector[:]
          tmp0 = XX.shape[0]
          tmp1 = XX.shape[1]
          XX = np.concatenate(XX,zVector)
          XX = np.reshape(XX,(tmp0+1,tmp1+2*n_L))

          spsi= math.sin(self.XX[8]);
          cpsi= math.cos(self.XX[8]);
          for i in range(n_L):
              ind= np.arange((15 + (2*i-1)),(15 + 2*i))
    
              dx= self.XX[ind[1]] - self.XX[1];
              dy= self.XX[ind[1]] - self.XX[1];
    
              H= np.array([[-cpsi, -spsi, -dx*spsi + dy*cpsi],[spsi,  -cpsi, -dx*cpsi - dy*spsi]]) 
              Y= H * self.PX[0:1,8] * np.transpose(H) + R
              
              tmp0 = PX.shape[0]
              tmp1 = PX.shape[1]
              PX = np.concatenate(PX,Y)
              PX = np.reshape(PX,(tmp0+1,tmp1+2))
      # ----------------------------------------------
      # ----------------------------------------------    
      def increase_landmarks_cov(self, minPXLM):

          if ((self.PX).shape[0] == 15): 
             return 0 
          PXLM= np.diag( self.PX[15:,15:] )
          minPXLM= minPXLM * np.ones((PXLM.shape[0],1));
          newDiagLM= max(PXLM,minPXLM);
          diffDiagLM= PXLM - newDiagLM;
          self.PX[15:,15:]= self.PX[15:end,15:end] - np.diag( diffDiagLM )

# =====================================================================================  
def discretize(F, G, S, dT):
    #MATRICES2DISCRETE This def discretize the continuous time model. It
    #works for either the GPS or IMU discretization times.
    # sysc= ss(F, zeros(15,1), zeros(1,15), 0)
    # sysd= c2d(sysc, dT)
    # Phi= sysd.A

    # Methdo to obtain covariance matrix for dicrete system
    C= np.transpose([[-F, G*S*np.transpose(G)],[zeros(15), np.transpose(F)]])

    # Proper method
    EXP= expm[C*dT]
    Phi= np.transpose(EXP[15:,15:])
    D_bar= Phi * EXP[1:14,15:]

    # Simplified method
    D_bar= (G*dT) * (S/dT) * np.transpose((G*dT)) # simplified version

    return [Phi, D_bar]

 # ----------------------------------------------
 # ----------------------------------------------  
def yawMeasurement(w,r_IMU2rearAxis):
    global XX

    r= np.array([[-r_IMU2rearAxis],[0],[0]])
    v_o= XX[3:5]
    R_NB= R_NB_rot.R_NB_rot(XX[6],XX[7],XX[8])
    v_a= v_o + R_NB * np.cross(w,r)
    v_a= v_a / np.norm(v_a)
    yaw= math.atan2(v_a[1],v_a[0]) 
    return yaw
 # ----------------------------------------------
 # ----------------------------------------------       


