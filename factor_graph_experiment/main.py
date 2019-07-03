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
import integrityMonitoringClassFgExpOff
import EstimatorClassFgExpOff



# create objects
params= ParametersClass("experiment_fg_offline");
load([params.path, 'FG.mat']); # organized experimental data (preprocessing using KF)
estimator= EstimatorClassFgExpOff.EstimatorClassFgExpOff(params);
im= integrityMonitoringClassFgExpOff.IntegrityMonitoringClassFgExpOff(params, estimator);
data_obj= DataClass(length(FG.imu), length(FG.lidar), params);
counters= CountersClass(np.array([]), np.array([]), params);


# ----------------------------------------------------------
# -------------------------- LOOP --------------------------
for epoch in range( 0,length(FG.imu) ):
    print('Epoch -> '+str(epoch))
    # set the simulation time to the IMU time
    counters.time_sim= counters.time_sim + 1/10;
    
    # Update the current state vector using the preprocessed data
    estimator.XX= FG.pose{epoch};
    estimator.x_true= FG.pose{epoch};
    
    # build the process noise and state evolution jacobian for IMU
    estimator.compute_imu_Phi_k( params, FG, epoch );
    
    # build the whiten jacobian for landmarks in the field of view
    estimator.compute_lidar_H_k( params, FG, epoch );
    
    # build the whiten jacobian for GPS msmt
    estimator.compute_gps_H_k( params, FG, epoch );
    
    # main function for factor graphs integrity monitoring
    im.monitor_integrity( estimator, counters, data_obj,  params );
    
    # Store data
    counters.k_update= data_obj.store_update_fg(counters.k_update, estimator, counters.time_sim, params);

    # increase integrity counter
    counters.increase_integrity_monitoring_counter();
    

'''
# -------------------------- PLOTS --------------------------
data_obj.plot_map_localization_sim(estimator, params.num_epochs_sim, params)
data_obj.plot_number_of_landmarks_fg_sim(params);
data_obj.plot_integrity_risk(params);
#------------------------------------------------------------'''
