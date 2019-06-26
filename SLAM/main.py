import numpy as np
import math
import sys
sys.path.insert(0,'../utils/functions/')
sys.path.insert(0,'../utils/class/')
import ParametersClass
import EstimatorClassSlam
import CountersClass
import LidarClass
#import DataClass
import GPSClass
import LidarClass
import IMUClass
import DataClass

# create objects
params= ParametersClass.ParametersClass('slam');
gps= GPSClass.GPSClass(params.num_epochs_static * params.dt_imu, params);
lidar= LidarClass.LidarClass(params,gps.timeInit);
#imu= IMUClass.IMUClass(params, 70478);
#estimator= EstimatorClassSlam(imu.inc_msmt(1:3, params.num_epochs_static), params);
data_obj= DataClass.DataClass(302, gps.num_readings, params);#imu.num_readings
counters= CountersClass.CountersClass(gps, lidar, params);
