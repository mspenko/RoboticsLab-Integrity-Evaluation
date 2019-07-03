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
import IntegrityMonitoringClassEkfSim
import EstimatorClassEkfSim



# create objects
params= ParametersClass.ParametersClass("simulation_kf");
#im= IntegrityMonitoringClassEkfSim.IntegrityMonitoringClassEkfSim(params);
estimator= EstimatorClassEkfSim.EstimatorClassEkfSim(params);
data_obj= DataClass.DataClass(params.num_epochs_sim, params.num_epochs_sim, params);
counters= CountersClass.CountersClass(np.array([]), np.array([]), params);

# ----------------------------------------------------------
# -------------------------- LOOP --------------------------
for epoch in range( 1,params.num_epochs_sim+1):
    print('Epoch -> '+str(epoch))
    # ------------- Odometry -------------
    estimator.odometry_update( params );
    # -------------------------------
    
    # Store data
    data_obj.store_prediction_sim(epoch, estimator, counters.time_sim);
    
    # ------------------- GPS -------------------
    if (params.SWITCH_GPS_UPDATE and counters.time_sum >= params.dt_gps):
        # GPS update 
        z_gps= estimator.get_gps_msmt(params);
        estimator.gps_update( z_gps, params );
        
        # save GPS measurements
        data_obj.store_gps_msmts(z_gps);
        
        # reset counter for GPS
        counters.reset_time_sum();
    end
    # ----------------------------------------
    
    # ----------------- LIDAR ----------------
    if (params.SWITCH_LIDAR_UPDATE == 1):
        
        # Simulate lidar feature measurements
        z_lidar= estimator.get_lidar_msmt( params );
        
        # NN data association
        estimator.nearest_neighbor(z_lidar, params);
        
#         # introduce miss-association
#         if estimator.x_true(1) < 70.5
#             for i= 1:length(estimator.association_true)
#                 if estimator.association_true(i) == 4
#                     estimator.association(i)= 6;
#                 elseif estimator.association_true(i) == 6
#                     estimator.association(i)= 0;
#                 elseif estimator.association_true(i) == 5
#                     estimator.association(i)= 0;
#                 elseif estimator.association_true(i) == 7
#                     estimator.association(i)= 5;
#                 end
#             end
#         end

        # Evaluate the probability of mis-associations
        im.prob_of_MA( estimator, params);
        im.P_MA_k= np.array([0]);
        
        # Lidar update
        estimator.lidar_update(z_lidar, params);
        
        # integrity monitoring
        im.monitor_integrity(estimator, counters, data_obj, params);
        
        # Add current msmts in Nav-frame
        data_obj.store_msmts( body2nav_2D(z_lidar, estimator.XX, estimator.XX[2]) ); 
        
        # Store data
        counters.k_update=data_obj.store_update_sim(counters.k_update, estimator, counters.time_sim, params);
        
        # increase integrity counter
        counters.increase_integrity_monitoring_counter();
    end
    # -----------------------------------------
    
    # increase time
    counters.increase_time_sum_sim(params);
    counters.increase_time_sim(params);
end
# ------------------------- END LOOP -------------------------
# ------------------------------------------------------------




# Store data for last epoch
data_obj.delete_extra_allocated_memory(counters)

'''
# ------------- PLOTS -------------
data_obj.plot_map_localization_sim(estimator, params.num_epochs_sim, params)
data_obj.plot_number_of_landmarks(params);
# data_obj.plot_number_epochs_in_preceding_horizon(params);
# data_obj.plot_estimates();
data_obj.plot_integrity_risk(params);
# data_obj.plot_MA_probabilities();
data_obj.plot_error(params);
# data_obj.plot_P_H();
data_obj.plot_detector(params);
# ------------------------------------------------------------'''
