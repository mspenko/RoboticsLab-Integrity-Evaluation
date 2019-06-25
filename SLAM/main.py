import numpy as np
import math
import sys
sys.path.insert(0,'../utils/functions/')
sys.path.insert(0,'../utils/class/')
import ParametersClass
import CountersClass
import DataClass

# create objects
params= ParametersClass.ParametersClass("simulation_kf")
#estimator= EstimatorClassEkfSim(params);
data_obj= DataClass.DataClass(params.num_epochs_sim, params.num_epochs_sim, params)
counters= CountersClass.CountersClass([], [], params)
