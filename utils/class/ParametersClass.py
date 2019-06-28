import yaml
import numpy as np
from scipy.linalg import block_diag
from scipy.linalg import sqrtm
from scipy.linalg import inv
from scipy.stats.distributions import chi2
import math


class ParametersClass:
 # ----------- Switches (options) ---------------
      SWITCH_REDUCE_TESTING = None
      SWITCH_VIRT_UPDATE_Z = None
      SWITCH_VIRT_UPDATE_Y = None
      SWITCH_YAW_UPDATE = None
      SWITCH_GPS_UPDATE = None
      SWITCH_GPS_VEL_UPDATE = None
      SWITCH_LIDAR_UPDATE = None
      SWITCH_REMOVE_FAR_FEATURES = None
      SWITCH_CALIBRATION = None
      SWITCH_FIXED_LM_SIZE_PH = None
      SWITCH_LM_SELECTION = None
      SWITCH_SEED = None
      SWITCH_ONLY_ONE_LM_FAULT = None
      SWITCH_GENERATE_RANDOM_MAP = None
      SWITCH_LIDAR_FAULTS = None
      SWITCH_GPS_FG = None
      SWITCH_FIXED_ABS_MSMT_PH_WITH_min_GPS_msmt = None

      SWITCH_SLAM= 0
      SWITCH_SIM= 0
      SWITCH_FACTOR_GRAPHS= 0
      SWITCH_OFFLINE= 0

      path_test= '../../data/cart/20180725/'
      path_test= '../data/vehicle/20190110/'
      path_sim_kf= '../data/simulation/straight/'
      path_sim= '../../data/simulation/square/'
      path_sim_fg= '../../data/simulation/factor_graph/'
      path_exp_fg= '../../data/vehicle/20190110/'

# ---------------------------------------------------------
# ------------ paramters specified in file ----------------
# ---------------------------------------------------------

      m = None
      I_MA = None
      P_MA_max = None
      P_UA = None
      I_H = None
      min_appearances = None
      num_epochs_reduce_testing = None
      num_epochs_static = None
      lidarRange = None
      m_F = None
      dt_imu = None
      dt_cal = None
      dt_virt_z = None
      dt_virt_y = None
      sig_cal_pos = None
      sig_cal_vel = None
      sig_cal_E = None
      sig_yaw0 = None
      sig_phi0 = None
      sig_ba = None
      sig_bw = None
      sig_virt_vz = None
      sig_virt_vy = None
      sig_lidar = None
      min_vel_gps = None
      min_vel_yaw = None
      taua_normal_operation = None
      tauw_normal_operation = None
      taua_calibration = None
      tauw_calibration = None
      g_val = None
      r_IMU2rearAxis = None
      alpha_NN = None
      threshold_new_landmark = None
      sig_minLM = None
      mult_factor_acc_imu = None
      mult_factor_gyro_imu = None
      mult_factor_pose_gps = None
      mult_factor_vel_gps = None
      feature_height = None
      initial_yaw_angle = None
      preceding_horizon_size = None
      M = None # same as preceding_horizon_size = None
      min_n_L_M = None
      continuity_requirement = None
      alert_limit = None
      VRW = None
      ARW = None
      sn_f = None
      sn_w = None

   # -------------------- simulation -----------------------

      num_epochs_sim = None
      dt_sim = None
      dt_gps = None
      velocity_sim = None
      steering_angle_sim = None
      sig_gps_sim = None
      sig_velocity_sim = None
      sig_steering_angle_sim = None
      R_gps_sim = None
      W_odometry_sim = None
      wheelbase_sim = None

   # -------------------------------------------
   # -------------------------------------------

      path = None
      file_name_imu = None
      file_name_gps = None
      file_name_lidar_path = None
      file_name_calibration = None

      ind_pose = None # indexes that extract x-y-theta
      ind_yaw = None # index that extracts theta

      g_N = None # G estimation (sense is same at grav acceleration in nav-frame)
      sig_cal_pos_blkMAtrix = None
      sig_cal_vel_blkMAtrix = None
      sig_cal_E_blkMAtrix = None
      R_cal = None
      H_cal = None # Calibration observation matrix
      H_yaw = None
      R_virt_Z = None
      R_virt_Y = None
      R_lidar = None
      sqrt_inv_R_lidar = None
      T_NN = None
      xPlot = None
      yPlot = None
      zPlot = None
      xyz_B = None
      R_minLM = None

   # IMU -- white noise specs

      sig_IMU_acc = None
      sig_IMU_gyr = None
      V = None
      Sv = None     # PSD for IMU
      Sv_cal = None # PSD during calibration

   # Biases -- PSD of white noise
      Sn_f = None # TODO: check if needed
      Sn_w = None # TODO: check if needed
      Sn = None

   # PSD for continuous model
      S = None
      S_cal = None

   # ------------------ Factor Graphs ---------------------
      way_points = None
      min_distance_to_way_point = None
      max_delta_steering = None #maximum change in steering angle during one second
      max_steering = None
      velocity_FG =  None
      FG_prec_hor = None
      sig_velocity_FG = None
      sig_steering_angle_FG = None
      W_odometry_FG = None
      wheelbase_FG =  None
      min_state_var_FG =  None
      sig_gyro_z =  None
      map_limits = None # [x_min, x_max, y_min,  y_max]
      optimoptions = None # optimoptions for the fg optimization
      landmark_density = None # landmarks / m^2
      landmark_map = None
   # -------------------------------------------
   # -------------------------------------------
          
      def __init__(self,navigation_type):
        
           if(navigation_type == 'slam'):
              self.SWITCH_SLAM= 1
              self.path= self.path_test

           elif(navigation_type == 'localization_kf'):
              self.path = self.path_test

           elif(navigation_type == 'localization_fg'):
              self.SWITCH_FACTOR_GRAPHS= 1
              self.path = self.path_test

           elif(navigation_type == 'simulation_kf'):
              self.SWITCH_SIM= 1
              self.path= self.path_sim_kf
              
           elif(navigation_type == 'simulation_fg_offline'):

              self.SWITCH_SIM= 1   
              self.SWITCH_FACTOR_GRAPHS= 1   
              self.SWITCH_OFFLINE= 1   
              self.path= self.path_sim_fg   

           elif(navigation_type == 'simulation_fg_online'):
              self.SWITCH_SIM= 1   
              self.SWITCH_FACTOR_GRAPHS= 1   
              self.path= self.path_sim_fg   

           elif(navigation_type == 'experiment_fg_offline'):
              self.SWITCH_SIM= 0   
              self.SWITCH_FACTOR_GRAPHS= 1   
              self.SWITCH_OFFLINE= 1   
              self.path= self.path_exp_fg   
           else:
               print("[error] NAVIGATION_TYPE MUST BE EITHER: 'slam', 'localization' OR 'factor_graph'")


            # ---------------------------------------------------------  
            # ------------ paramters specified in file ----------------
            # ---------------------------------------------------------

           tmp_path = self.path + "parameters.yaml"
           in_file  = open(tmp_path)
           para     = yaml.load(in_file)
 
           
            # --------------- Switches (options) ---------------
           self.SWITCH_REDUCE_TESTING = para['SWITCH_REDUCE_TESTING']
           self.SWITCH_VIRT_UPDATE_Z = para['SWITCH_VIRT_UPDATE_Z']
           self.SWITCH_VIRT_UPDATE_Y = para['SWITCH_VIRT_UPDATE_Y']
           self.SWITCH_YAW_UPDATE = para['SWITCH_YAW_UPDATE']
           self.SWITCH_GPS_UPDATE = para['SWITCH_GPS_UPDATE']
           self.SWITCH_GPS_VEL_UPDATE = para['SWITCH_GPS_VEL_UPDATE']
           self.SWITCH_LIDAR_UPDATE = para['SWITCH_LIDAR_UPDATE']
           self.SWITCH_REMOVE_FAR_FEATURES = para['SWITCH_REMOVE_FAR_FEATURES']
           self.SWITCH_CALIBRATION = para['SWITCH_CALIBRATION']
           self.SWITCH_FIXED_LM_SIZE_PH = para['SWITCH_FIXED_LM_SIZE_PH']
           self.SWITCH_LM_SELECTION = para['SWITCH_LM_SELECTION']
           self.SWITCH_SEED = para['SWITCH_SEED']
           self.SWITCH_ONLY_ONE_LM_FAULT = para['SWITCH_ONLY_ONE_LM_FAULT']


           if self.SWITCH_FACTOR_GRAPHS == 1:
              self.SWITCH_GENERATE_RANDOM_MAP = para['SWITCH_GENERATE_RANDOM_MAP']
              self.SWITCH_LIDAR_FAULTS = para['SWITCH_LIDAR_FAULTS']
              if(self.SWITCH_SIM == 0):
                self.SWITCH_GPS_FG = para['SWITCH_GPS_FG']
                self.SWITCH_FIXED_ABS_MSMT_PH_WITH_min_GPS_msmt = para['SWITCH_FIXED_ABS_MSMT_PH_WITH_min_GPS_msmt']

           self.m = para['m']
           self.I_MA = para['I_MA']
           self.P_MA_max = para['P_MA_max']
           self.P_UA = para['P_UA']
           self.I_H = para['I_H']
           self.min_appearances = para['min_appearances']
           self.num_epochs_reduce_testing = para['num_epochs_reduce_testing']
           self.num_epochs_static = para['num_epochs_static']
           self.lidarRange = para['lidarRange']
           self.m_F = para['m_F']
           self.dt_imu = eval(para['dt_imu'])
           self.dt_cal = eval(para['dt_cal'])
           self.dt_virt_z = eval(para['dt_virt_z'])
           self.dt_virt_y = eval(para['dt_virt_y'])
           self.sig_cal_pos = para['sig_cal_pos']
           self.sig_cal_vel = para['sig_cal_vel']
           self.sig_cal_E = np.deg2rad(para['sig_cal_E'])
           self.sig_yaw0 = np.deg2rad(para['sig_yaw0'])
           self.sig_phi0 = np.deg2rad(para['sig_phi0'])
           self.sig_ba = para['sig_ba']
           self.sig_bw = eval(para['sig_bw'])
           self.sig_virt_vz = para['sig_virt_vz']
           self.sig_virt_vy = para['sig_virt_vy']
           self.sig_lidar = para['sig_lidar']
           self.min_vel_gps = eval(para['min_vel_gps'])
           self.min_vel_yaw = eval(para['min_vel_yaw'])
           self.taua_normal_operation = para['taua_normal_operation']
           self.tauw_normal_operation = para['tauw_normal_operation']
           self.taua_calibration = para['taua_calibration']
           self.tauw_calibration = para['tauw_calibration']
           self.g_val = para['g_val']
           self.r_IMU2rearAxis = para['r_IMU2rearAxis']
           self.alpha_NN = para['alpha_NN']
           self.threshold_new_landmark = para['threshold_new_landmark']
           self.sig_minLM = para['sig_minLM']
           self.mult_factor_acc_imu = para['mult_factor_acc_imu']
           self.mult_factor_gyro_imu = para['mult_factor_gyro_imu']
           self.mult_factor_pose_gps = para['mult_factor_pose_gps']
           self.mult_factor_vel_gps = para['mult_factor_vel_gps']
           self.feature_height = para['feature_height']
           self.initial_yaw_angle = para['initial_yaw_angle']
           self.preceding_horizon_size = para['preceding_horizon_size']
           self.M = para['preceding_horizon_size']
           self.continuity_requirement = para['continuity_requirement']
           self.alert_limit = para['alert_limit']
           self.VRW = para['VRW']
           self.ARW = para['ARW']
           self.sn_f = eval(para['sn_f'])
           self.sn_w = eval(para['sn_w'])
           self.min_n_L_M = para['min_n_L_M']
           #--------------------simulation-------------------------

           if (self.SWITCH_SIM == 1):
              self.num_epochs_sim= para['num_epochs_sim']
              self.dt_sim= para['dt_sim']
              self.dt_gps= para['dt_gps']
              self.velocity_sim= para['velocity_sim']
              self.steering_angle_sim= para['steering_angle_sim']
              self.sig_gps_sim= para['sig_gps_sim']
              self.sig_velocity_sim= para['sig_velocity_sim']
              self.sig_steering_angle_sim= para['sig_steering_angle_sim']
              self.wheelbase_sim= para['wheelbase_sim']
              # if using factor graphs that needs controller
              if (self.SWITCH_FACTOR_GRAPHS == 1):
                    self.way_points= para['way_points']
                    self.min_distance_to_way_point= para['min_distance_to_way_point']
                    self.max_delta_steering= para['max_delta_steering']
                    self.max_steering= para['max_steering']
                    self.sig_gyro_z= para['sig_gyro_z']
                    self.map_limits= para['map_limits']
                    self.landmark_density= para['landmark_density']


           # -------------------------------------------
           # -------------------------------------------
            
           # set file names
           self.file_name_imu= self.path + 'IMU/IMU.mat'
           self.file_name_gps= self.path + 'GPS/GPS.mat'
           self.file_name_lidar_path= self.path + 'LIDAR/'
           self.file_name_calibration= self.path + 'IMU/calibration.mat'

           # modify parameters
           self.VRW= self.VRW * self.mult_factor_acc_imu 
           self.ARW= self.ARW * self.mult_factor_gyro_imu #######################  CAREFUL

           # ------------------ build parameters ------------------

           if (self.SWITCH_SEED == 1):
               self.set_seed_to(para['SWITCH_SEED'])
           else:
              None

           if(self.SWITCH_SIM == 1):
                self.ind_pose= np.array([0,1,2])
                self.ind_yaw= 3
           else:
                self.ind_pose = np.array([1,2,9])
                self.ind_yaw = 9

           self.g_N= np.array([[0], [0], [self.g_val]]) # G estimation (sense is same at grav acceleration in nav-frame)
           self.sig_cal_pos_blkMAtrix= np.diag(np.array([self.sig_cal_pos, self.sig_cal_pos, self.sig_cal_pos]))
           self.sig_cal_vel_blkMAtrix= np.diag(np.array([self.sig_cal_vel, self.sig_cal_vel, self.sig_cal_vel]))
           self.sig_cal_E_blkMAtrix= np.diag(np.array([self.sig_cal_E, self.sig_cal_E, self.sig_cal_E]))

           self.R_cal=block_diag(self.sig_cal_pos_blkMAtrix, self.sig_cal_vel_blkMAtrix, self.sig_cal_E_blkMAtrix)**2
           self.H_cal= [np.eye(9), np.zeros((9,6))]# Calibration observation matrix
           self.H_yaw= [np.zeros((1,8)),1,np.zeros((1,6))]
           self.R_virt_Z= self.sig_virt_vz**2
           self.R_virt_Y= self.sig_virt_vy**2
           self.R_lidar= np.diag( [self.sig_lidar, self.sig_lidar] )**2
           self.sqrt_inv_R_lidar= sqrtm( inv( self.R_lidar ) )
           self.T_NN= chi2.ppf(1-self.alpha_NN,df=2) #chi2inv
           xPlot= np.array([[-0.3],[0], [-0.3]])
           yPlot=np.array( [[0.1], [0], [-0.1]])
           zPlot= np.array([[0], [0], [0]])
           self.xyz_B= np.transpose(np.array([xPlot, yPlot, zPlot]));
           self.R_minLM= self.sig_minLM**2;

           # IMU -- white noise specs      
           self.sig_IMU_acc= self.VRW * math.sqrt( 2000 / 3600 );
           self.sig_IMU_gyr= np.deg2rad( self.ARW * math.sqrt( 2000 / 3600 ) ); # rad
           self.V= np.diag( [self.sig_IMU_acc, self.sig_IMU_acc, self.sig_IMU_acc,self.sig_IMU_gyr, self.sig_IMU_gyr, self.sig_IMU_gyr])**2;       

           self.Sv= self.V * float(self.dt_imu); # Convert to PSD
           self.Sv_cal= np.diag( [np.diag( self.Sv[0:3,0:3]) / self.mult_factor_acc_imu**2, np.diag(self.Sv[3:6,3:6]) / self.mult_factor_gyro_imu**2]  );

           # Biases -- PSD of white noise
           self.Sn_f= np.diag([self.sn_f, self.sn_f, self.sn_f]);
           self.Sn_w= np.diag([self.sn_w, self.sn_w, self.sn_w]);
           self.Sn= block_diag(self.Sn_f, self.Sn_w);
            
           # PSD for continuous model
           self.S     = block_diag(self.Sv, self.Sn);
           self.S_cal = block_diag(self.Sv_cal, self.Sn);

           '''
           if selfSWITCH_FACTOR_GRAPHS && ~selfSWITCH_OFFLINE
                selfoptimoptions= optimoptions( @fminunc,...
                    'Display', optimoptions_display,...
                    'Algorithm','trust-region',...
                    'SpecifyObjectiveGradient', true,...
                    'HessianFcn', 'objective');
                     'Algorithm','quasi-newton',...
            end'''
  
           # generate random map
           if (self.SWITCH_GENERATE_RANDOM_MAP ==1):
                self.landmark_map= self.return_random_map();

           # Debugging examples of landmark maps
           # obj.landmark_map= [-10.9771207481359,2.45300831544226;6.82608325798797,-73.1977929704449;89.5254275520182,-7.25453088645978;69.3398428399569,-48.8455972810412;114.138015445711,7.59971722452418;-6.44463501373658,-25.3416514918823;13.9192416318387,-14.3256046082598;-7.94271327641291,-7.48138926822496;29.6940122816621,37.7109503452478;14.1645971451474,29.7711855572875;74.3102246290761,17.3527677747590;-5.51789508821601,45.3269050389306;-6.49529972209515,65.4399001956805;99.6596174159157,-39.2371589565058;40.0752853981384,3.60822696762624;89.9944937044295,69.2199345817613;-18.4833642650074,-0.955765466280809;11.0461954904917,-20.7381963865714;91.7682610078083,52.8560314670443;87.0765301960277,22.0728675799338;-4.48118755183638,-65.9624870567692;2.09681884055983,23.8958079086734;-22.0400992168105,-61.1069502958662];
           # obj.landmark_map= [-19.2774312209662,40.7797185222247;59.5705993506951,68.4223427974766;117.850869342007,49.6705256737122;60.1459688918045,-4.29842354000353;101.819694216991,13.8251422010301;-12.5688746372810,10.3136104813153;-13.9277636518960,-20.5550507384034;-22.7018267455224,-6.84870330092294;51.3962526676511,-39.2466178828560;96.4565013671369,20.4352206248171;65.5377942259571,-47.5557574518067;32.5792016371773,-74.3808199744421;73.4539646924914,2.74001094196981;89.0013294523708,18.5805534077841;27.4379300947862,-20.3596935284201;66.5356210683029,48.5736256997634;9.57781595754895,71.3038070336353;83.0186438364574,-21.1886649971006;111.193736810766,-56.4883691287973;95.8978125038271,1.75304139073606;97.2231654030829,70.5468329863530;107.770973263735,65.2853361241904;44.8379555623654,38.0575870723817];
           # obj.landmark_map= [114.561090901347,-8.69890651128443;90.8910816769900,-20.0852116676695;14.5249429017169,20.7628311222522;58.7228710109234,39.0004333523196
           
            
      #------------------------------------------------------
      #------------------------------------------------------
      def set_seed_to(self, seed):
           np.random.seed(seed)
      #------------------------------------------------------
      #------------------------------------------------------
      def return_random_map(self):
          x_dim= self.map_limits(2) - self.map_limits(1);
          y_dim= self.map_limits(4) - self.map_limits(3);
          map_area= x_dim * y_dim;
          num_landmarks= round(selflandmark_density * map_area);
          self.landmark_map= [np.random.rand( num_landmarks, 1 ) * x_dim + self.map_limits(1),
                           np.random.rand( num_landmarks, 1 ) * y_dim + self.map_limits(3)];
      #------------------------------------------------------
      #------------------------------------------------------
      def sig_yaw_fn(self, v):
          sig_yaw= float(np.deg2rad(5) + ( np.exp(10*v)-1 )**(-1)); #6.6035
          return sig_yaw
           ############################ CAREFUL
      #------------------------------------------------------
      #------------------------------------------------------
      def R_yaw_fn(self, v):
          R_yaw= self.sig_yaw_fn(v)**2;
          return R_yaw 
          
      #------------------------------------------------------
      #------------------------------------------------------
      def turn_off_calibration(self):
          self.SWITCH_CALIBRATION= 0;
      #------------------------------------------------------
      #------------------------------------------------------
      def turn_off_lidar(self):
          self.SWITCH_LIDAR_UPDATE= 0;
      #------------------------------------------------------
      #------------------------------------------------------                       
      def turn_off_gps(self):
            self.SWITCH_GPS_UPDATE= 0;

 
        
