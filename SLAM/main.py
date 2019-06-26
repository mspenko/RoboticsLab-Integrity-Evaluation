import numpy as np
import math
import sys
sys.path.insert(0,'../utils/functions/')
sys.path.insert(0,'../utils/class/')
import ParametersClass
import EstimatorClassSlam
import CountersClass
import LidarClass
import GPSClass
import LidarClass
import IMUClass
import DataClass

# create objects
params= ParametersClass.ParametersClass('slam');
gps= GPSClass.GPSClass(params.num_epochs_static * params.dt_imu, params);
lidar= LidarClass.LidarClass(params,gps.timeInit);
imu= IMUClass.IMUClass(params, gps.timeInit);
estimator=  EstimatorClassSlam.EstimatorClassSlam(imu.inc_msmt[0:3, params.num_epochs_static], params);
data_obj= DataClass.DataClass(302, gps.num_readings, params);#imu.num_readings
counters= CountersClass.CountersClass(gps, lidar, params);

GPS_Index_exceeded = 0;   # TODO: osama is this needeed?
LIDAR_Index_exceeded = 0; # TODO: osama is this needeed?

# Initial discretization for cov. propagation
estimator.linearize_discretize( imu.msmt[:,0], params.dt_imu, params );

# ----------------------------------------------------------
# -------------------------- LOOP --------------------------
for epoch in range(imu.num_readings-1):
    print('Epoch -> ',epoch)
    # set the simulation time to the IMU time
    counters.time_sim= imu.time[epoch];

    # Turn off GPS updates if start moving
    if (epoch == params.num_epochs_static):
        params.turn_off_calibration();
        estimator.PX[6,6]= params.sig_phi0**2;
        estimator.PX[7,7]= params.sig_phi0**2;
        estimator.PX[8,8]= params.sig_yaw0**2;

    # Increase time count
    counters.increase_time_sums(params);   
    # ------------- IMU -------------
    estimator.imu_update( imu.msmt[:,epoch], params );
    # -------------------------------
    # Store data
    data_obj.store_prediction(epoch, estimator, counters.time_sim);

    # Increase time count
    counters.increase_time_sums(params);
    
    # ------------- IMU -------------
    estimator.imu_update( imu.msmt[:,epoch], params );
    # -------------------------------
    
    # Store data
    data_obj.store_prediction(epoch, estimator, counters.time_sim);
    # ------------------------------------
    
    # ------------- virtual msmt update >> Z vel  -------------  
    if (counters.time_sum_virt_z >= params.dt_virt_z and params.SWITCH_VIRT_UPDATE_Z==1 and params.SWITCH_CALIBRATION==0):
        zVelocityUpdate( params.R_virt_Z );
        counters.reset_time_sum_virt_z();
    # ---------------------------------------------------------

    # ------------- virtual msmt update >> Y vel  -------------  
    if( counters.time_sum_virt_y >= params.dt_virt_y and params.SWITCH_VIRT_UPDATE_Y and ~params.SWITCH_CALIBRATION):
         
        # Yaw update
        if (params.SWITCH_YAW_UPDATE==1 and np.norm(estimator.XX[4:6]) > params.min_vel_yaw):
            print('yaw udpate');
            estimator.yaw_update( imu.msmt[3:5,epoch], params);
        counters.reset_time_sum_virt_y();
    # ---------------------------------------------------------

    # ------------------- GPS -------------------
    if ((counters.time_sim + params.dt_imu) > counters.time_gps and GPS_Index_exceeded == 0):
        
        if (params.SWITCH_CALIBRATION==0 and params.SWITCH_GPS_UPDATE==1):
            # GPS update -- only use GPS vel if it's fast
            estimator.gps_update( gps.msmt[:,counters.k_gps], gps.R[:,counters.k_gps], params);
            
            # Yaw update
            if (params.SWITCH_YAW_UPDATE and norm(estimator.XX[3:5]) > params.min_vel_yaw):
                print('yaw udpate');
                estimator.yaw_update( imu.msmt[3:5,epoch], params );
            #estimator.linearize_discretize( imu.msmt[:,epoch], params.dt_imu, params);

            # Store data
            counters.k_update= data_obj.store_update( counters.k_update, estimator, counters.time_sim );
        
        # Time GPS counter
        counters.increase_gps_counter();
        
        # -----Osama----- TODO: Osama what is this??
        if (counters.k_gps <= gps.time.shape[1]):
            counters.time_gps= gps.time[counters.k_gps];
        else:
           counters.k_gps = counters.k_gps -1 ;
           GPS_Index_exceeded = 1;
    # ----------------------------------------
     # ------------- LIDAR -------------
    if ((counters.time_sim + params.dt_imu) > counters.time_lidar and params.SWITCH_LIDAR_UPDATE==1):
        
        if (epoch > 2000): #params.num_epochs_static - 3000
            # Read the lidar features
            epochLIDAR= lidar.time(counters.k_lidar,1);
            lidar.get_msmt( epochLIDAR, params );
            
            # Remove people-features for the data set
            lidar.remove_features_in_areas(estimator.XX[0:8]);
            
            # NN data association
            association= estimator.nearest_neighbor(lidar.msmt[:,1:2], params);
            
            # Lidar update
            estimator.lidar_update(lidar.msmt[:,1:2], association, params);

            # Increase landmark covariance to the minimum
            estimator.increase_landmarks_cov(params.R_minLM);
            
            # Add new landmarks
            estimator.addNewLM( lidar.msmt[np.transpose(association) == -1,:], params.R_lidar );
            
            # Lineariza and discretize
            estimator.linearize_discretize( imu.msmt[:,epoch], params.dt_imu, params);
            
            # Store data
            data_obj.store_msmts( body2nav_3D(lidar.msmt, estimator.XX[1:9]) );# Add current msmts in Nav-frame
            counters.k_update= data_obj.store_update(counters.k_update, estimator, counters.time_sim);
        end
        
        # Increase counters
        counters.increase_lidar_counter();
        
        # -----Osama----- TODO: osama, what is this again?
        if (counters.k_lidar <= length(lidar.time)):
            counters.time_lidar= lidar.time(counters.k_lidar,2);
        else:
           counters.k_lidar = counters.k_lidar -1 ;
           LIDAR_Index_exceeded = 1;
