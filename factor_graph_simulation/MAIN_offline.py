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

for map_i in range(10):
    
    # seed the randomness
    np.random.random(map_i)    
    # create objects
    params= ParametersClass.ParametersClass("simulation_fg_offline");
    estimator= EstimatorClassFgSimOff.EstimatorClassFgSimOff(params);
    im= IntegrityMonitoringClassFgSimOff.IntegrityMonitoringClassFgSimOff(params, estimator);
    data_obj= DataClass.DataClass(params.num_epochs_sim, params.num_epochs_sim, params);
    counters= CountersClass.CountersClass(np.array([]), np.array([]), params);

    # initialize time index
    epoch= 1;

    # ----------------------------------------------------------
    # -------------------------- LOOP --------------------------
    acc = 0
    while (estimator.goal_is_reached==0 and epoch <= params.num_epochs_sim):
           print('Epoch -> '+str(epoch))
           # ------------- Odometry -------------
           estimator.compute_steering(params)
           estimator.odometry_update(params);
           # -------------------------------
    
           # ----------------- LIDAR ----------------
           if (params.SWITCH_LIDAR_UPDATE == 1):

                # build the jacobian landmarks in the field of view
                estimator.compute_lidar_H_k( params );
            
                # main function for factor graphs integrity monitoring
                im.monitor_integrity(estimator, counters, data_obj,  params);

                # Store data
                counters.k_update=data_obj.store_update_fg(counters.k_update, estimator, counters.time_sim, params);
         
                # increase integrity counter
                counters.increase_integrity_monitoring_counter();
           # -----------------------------------------
    
           # increase time
           counters.increase_time_sum_sim(params);
           counters.increase_time_sim(params);
           epoch= epoch + 1;
    # ------------------------- END LOOP -------------------------
    # ------------------------------------------------------------

    # Store data for last epoch
    data_obj.delete_extra_allocated_memory(counters)

    '''
    # save workspace
    save(strcat( params.path_sim_fg, 'results/density_001/map_', num2str(map_i), '/offline' ));

    end


       # -------------------------- PLOTS --------------------------
       data_obj.plot_map_localization_sim_fg(estimator, params)
       data_obj.plot_number_of_landmarks_fg_sim(params);
       data_obj.plot_integrity_risk(params);
       # ------------------------------------------------------------'''

