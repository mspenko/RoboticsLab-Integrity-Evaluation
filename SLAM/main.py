import numpy as np
import math
import sys
sys.path.insert(0,'../utils/functions/')
sys.path.insert(0,'../utils/class/')
import ParametersClass
import CountersClass
import LidarClass
#import DataClass
import GPSClass
import LidarClass
import IMUClass


# create objects
params= ParametersClass.ParametersClass('slam');
#gps= GPSClass.GPSClass(params.num_epochs_static * params.dt_imu, params);
lidar= LidarClass.LidarClass(params,70478) #gps.timeInit);
imu= IMUClass.IMUClass(params, 70478);
#estimator= EstimatorClassSlam(imu.inc_msmt(1:3, params.num_epochs_static), params);
#data_obj= DataClass(imu.num_readings, gps.num_readings, params);
#counters= CountersClass(gps, lidar, params);
