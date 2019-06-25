import numpy as np
import math
import sys
sys.path.insert(0,'../utils/functions/')
sys.path.insert(0,'../utils/class/')
import ParametersClass


# create objects
params= ParametersClass.ParametersClass("simulation_kf");
im= IntegrityMonitoringClassEkfSim(params);
#estimator= EstimatorClassEkfSim(params);
#data_obj= DataClass(params.num_epochs_sim, params.num_epochs_sim, params);
#counters= CountersClass([], [], params);
