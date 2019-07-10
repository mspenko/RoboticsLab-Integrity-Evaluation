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
import body2nav_3D
import IntegrityMonitoringClassEkfExp
import EstimatorClassEkfExp
import FGDataInputClass

# create objects
params= ParametersClass.ParametersClass("localization_kf");
gps= GPSClass.GPSClass(params.num_epochs_static * params.dt_imu, params);
lidar= LidarClass.LidarClass(params, gps.timeInit);
imu= IMUClass.IMUClass(params, gps.timeInit);
estimator= EstimatorClassEkfExp.EstimatorClassEkfExp(imu.msmt[0:3, 0:params.num_epochs_static], params);
im= IntegrityMonitoringClassEkfExp.IntegrityMonitoringClassEkfExp(params, estimator);
data_obj= DataClass.DataClass(imu.num_readings, lidar.num_readings, params);
counters= CountersClass.CountersClass(gps, lidar, params);
FG= FGDataInputClass.FGDataInputClass(lidar.num_readings);

# Initial discretization for cov. propagation
estimator.linearize_discretize( imu.msmt[:,0], params.dt_imu, params );

# ----------------------------------------------------------
# -------------------------- LOOP --------------------------
for epoch in range(imu.num_readings):
    print('Epoch -> '+str(epoch))

    # set the simulation time to the IMU time
    counters.time_sim= imu.time[epoch]

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
    
    # ------------- Calibration -------------
    if (counters.time_sum >= params.dt_cal and params.SWITCH_CALIBRATION == 1):
        
        # create a fake msmt and do a KF update to set biases 
        estimator.calibration(imu.msmt[:,epoch+1], params); #Osama 
        
        # Store data
        counters.k_update= data_obj.store_update(counters.k_update, estimator, counters.time_sim);
        counters.reset_time_sum();
    # ------------------------------------
    
    # ------------- virtual msmt update >> Z vel  -------------  
    if (counters.time_sum_virt_z >= params.dt_virt_z and params.SWITCH_VIRT_UPDATE_Z==1 and params.SWITCH_CALIBRATION==0):
        estimator.vel_update_z(params.R_virt_Z);
        counters.reset_time_sum_virt_z();

    # ---------------------------------------------------------
    
    # ------------- virtual msmt update >> Y vel  -------------
    if (counters.time_sum_virt_y >= params.dt_virt_y and params.SWITCH_VIRT_UPDATE_Y==1 and params.SWITCH_CALIBRATION==0):
         
        # Yaw update
        if (params.SWITCH_YAW_UPDATE and np.norm(estimator.XX[3:6]) > params.min_vel_yaw):
            print('yaw update');
            estimator.yaw_update( imu.msmt[3:6,epoch+1], params); #Osama
        counters.reset_time_sum_virt_y();
    # ---------------------------------------------------------
    
    
    # ------------------- GPS -------------------
    if ((counters.time_sim + params.dt_imu) > counters.time_gps):

        if ( (params.SWITCH_CALIBRATION==0) and (params.SWITCH_GPS_UPDATE==1) ):
            # GPS update -- only use GPS vel if it's fast
            estimator.gps_update( gps.msmt[:,counters.k_gps], gps.R[counters.k_gps,:], params);

            # This is used to store gps msmt and R recieved at lidar epoch for FG
            gps.IS_GPS_AVAILABLE= 1;
            current_gps_msmt= gps.msmt[:,counters.k_gps];
            current_gps_R= gps.R[counters.k_gps,:];

            # Yaw update
            if (params.SWITCH_YAW_UPDATE and np.linalg.norm(estimator.XX[3:6]) > params.min_vel_yaw):
                print('yaw update');
                estimator.yaw_update( imu.msmt[3:6,epoch+1], params ); #Osama
            estimator.linearize_discretize( imu.msmt[:,epoch+1], params.dt_imu, params); #Osama
            
            # Store data
            counters.k_update= data_obj.store_update( counters.k_update, estimator, counters.time_sim );

        # Time GPS counter
        if (counters.k_gps == gps.num_readings):
            params.turn_off_gps();
        else:
            counters.increase_gps_counter();
            counters.time_gps= gps.time[counters.k_gps];

    # ----------------------------------------
    
    # ------------- LIDAR -------------
    if ((counters.time_sim + params.dt_imu) > counters.time_lidar and params.SWITCH_LIDAR_UPDATE):
        
        if (epoch > params.num_epochs_static):

            # Read the lidar features
            epochLIDAR= lidar.time[counters.k_lidar,0];
            lidar.get_msmt( epochLIDAR, params );

            # Remove people-features for the data set
            lidar.remove_features_in_areas(estimator.XX[0:9]);

            # NN data association
            estimator.nearest_neighbor(lidar.msmt[:,0:2], params);

            # Evaluate the probability of mis-associations
            im.prob_of_MA( estimator, params);

            # Lidar update
            estimator.lidar_update(lidar.msmt[:,0:2], params);

            # Lineariza and discretize
            estimator.linearize_discretize( imu.msmt[:,epoch+1], params.dt_imu, params); #Osama
            
            # Store the required data for Factor Graph
            z= lidar.msmt[:,0:2];

            tmp_list = []
            for i in range(estimator.association.shape[0]):
                if (estimator.association[i] == -1):
                   tmp_list.append(i)
            z = np.delete(z, tmp_list, axis = 0)

            FG.lidar[counters.k_lidar]= z;
            FG.associations[counters.k_lidar]= estimator.association_no_zeros;
            FG.imu[counters.k_lidar]= imu.msmt[:,epoch];
            FG.pose[counters.k_lidar]= estimator.XX;
            if (gps.IS_GPS_AVAILABLE==1):
                FG.gps_msmt[counters.k_lidar]= current_gps_msmt;
                FG.gps_R[counters.k_lidar]= current_gps_R;
                gps.IS_GPS_AVAILABLE= 0; # mark the current gps reading as stored
            
            # integrity monitoring
            im.monitor_integrity(estimator, counters, data_obj, params);
            
            # Store data
            data_obj.store_msmts( body2nav_3D.body2nav_3D(lidar.msmt, estimator.XX[0:9]) );# Add current msmts in Nav-frame
            counters.k_update= data_obj.store_update(counters.k_update, estimator, counters.time_sim);
            # increase integrity counter
            counters.increase_integrity_monitoring_counter();
            
            
        else:
            
            # Index of last static lidar epoch(it will be obatined during the run)
            lidar.index_of_last_static_lidar_epoch= counters.k_lidar;

        
        # Time lidar counter
        if (counters.k_lidar == lidar.num_readings):
            params.turn_off_lidar; 
        else:
            counters.increase_lidar_counter();
            counters.time_lidar= lidar.time[counters.k_lidar,1];

# ------------------------- END LOOP -------------------------
# ------------------------------------------------------------




# Store data for last epoch
data_obj.store_update(counters.k_update, estimator, counters.time_sim);
data_obj.delete_extra_allocated_memory(counters)

# delete fields corresponding to static epochs
FG.delete_fields_corresponding_to_static_epochs(lidar)


'''
# ------------- PLOTS ------------
data_obj.plot_map_localization(estimator, gps, imu.num_readings, params)

data_obj.plot_number_of_landmarks(params);
data_obj.plot_number_epochs_in_preceding_horizon(params);
data_obj.plot_estimates();
data_obj.plot_integrity_risk(params);
data_obj.plot_MA_probabilities();'''
# data_obj.plot_P_H();
# --------------------------------
