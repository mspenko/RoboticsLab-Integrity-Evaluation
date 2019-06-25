import numpy as np
class PredictionDataClass:
      properties = None
      XX = None
      time = None
    

      def __init__(self,num_readings, params):
           # allocate memory
           self.XX= np.zeros((params.m, num_readings));
           self.time= np.zeros((num_readings, 1));
        
       # ----------------------------------------------
       # ----------------------------------------------
      def store(self, epoch, estimator, time):
          self.XX[:,epoch]= estimator.XX[1:15];
          self.time[epoch]= time;
        
       # ----------------------------------------------
       # ----------------------------------------------
      def store_sim(self, epoch, estimator, time):
          self.time[epoch]= time;   
          self.XX[:,epoch]= estimator.XX;
        
       # ----------------------------------------------
       # ----------------------------------------------
        
    




