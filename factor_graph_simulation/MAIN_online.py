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
import EstimatorClassFgSimOn


# create objects
params= ParametersClass.ParametersClass("simulation_fg_online");
estimator= EstimatorClassFgSimOn.EstimatorClassFgSimOn([], params);
data_obj= DataClass.DataClass(params.num_epochs_sim, params.num_epochs_sim, params);
counters= CountersClass.CountersClass([], [], params);


# ----------------------------------------------------------
# -------------------------- LOOP --------------------------
epoch= 1; # initialize time index

while (estimator.goal_is_reached==0 and epoch <= params.num_epochs_sim):

    print('Epoch -> '+str(epoch))
    # ------------- Odometry -------------
    estimator.compute_steering(params)
    estimator.odometry_update( params );
    # ------------------------------------
    
    # ------------- Gyro -------------
    if (epoch > 1):
        estimator.generate_gyro_msmt(data_obj.update.x_true[params.ind_yaw,epoch], estimator.x_true[params.ind_yaw-1], params);
    # --------------------------------
    
    
    # ----------------- LIDAR ----------------
     if (params.SWITCH_LIDAR_UPDATE ==1):

         # get the lidar msmts
         estimator.get_lidar_msmt(params);
         estimator.association= estimator.association_true;
                  
         # solve the fg optimization
         estimator.solve(counters, params)
         
         # update preceding horizon for the estimate
         estimator.update_preceding_horizon(params)
         
         # Store data
         counters.k_update= data_obj.store_update_fg(counters.k_update, estimator, counters.time_sim, params);
                  
         # increase counter
         counters.increase_lidar_counter();
         
    # -----------------------------------------
    
    # increase time
    counters.increase_time_sum_sim(params);
    counters.increase_time_sim(params);
    epoch= epoch + 1;
end
# ------------------------- END LOOP -------------------------
# ------------------------------------------------------------

# Store data for last epoch
data_obj.delete_extra_allocated_memory(counters)

'''
# -------------------------- PLOTS --------------------------
data_obj.plot_map_localization_sim_fg(estimator, params)
data_obj.plot_number_of_landmarks_fg_sim(params);
data_obj.plot_detector_fg(params);
data_obj.plot_error_fg(params);
# ------------------------------------------------------------'''

data_obj.find_HMI_sim(params)

