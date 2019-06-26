from scipy.linalg import block_diag
import numpy as np
import math

class IMUClass:
      num_readings = None
      time         = None
      inc_msmt     = None
      msmt         = None

      def __init__(self,params,initial_time):
         # DATAREAD Reads a datafile and outputs a series of vectors of the data
         # Data id in columns
 
         # load "data" for the IMU
         data = np.loadtxt(params.file_name_imu)
         g0= 9.80665 # m/s^2 -- by manufacturerw -- TODO: include in parameters
   
         self.time= data[:,3]
         gyroX   = np.deg2rad( data[:,4] )
         gyroY   = np.deg2rad( data[:,5] )
         gyroZ   = np.deg2rad( data[:,6] )
         # gyroSts = data[:][7]8)
         accX    = data[:,8]  *g0
         accY    = data[:,9] *g0
         accZ    = data[:,10] *g0
         # accSts  = data[:][11]
         incX    = data[:,12] *g0
         incY    = data[:,13] *g0
         incZ    = data[:,14] *g0
         # incSts  = data[:][15]

         # Use the first GPS reading time as reference
         self.time= self.time - initial_time
            
         # Set paramters
         self.inc_msmt= np.transpose(np.array([incX, incY, incZ]))
         self.msmt= np.transpose(np.array([accX, accY, accZ, gyroX, gyroY, gyroZ]))
            
         # num of readings
         self.num_readings= self.msmt.shape[1];
         
         invC  = np.array([[1.0010,0,0],[0,1.0066,0],[0,0,0.9958]])
         iinvC = np.array([[0.9993,0,0],[0,1.0045,0],[0,0,0.9949]])
         ib_0  = np.array([-0.0063,0.0690,-0.0377])
         b_0   = np.array([-0.0110,0.0265,-0.0442,-2.259871327935569e-04,1.150862443860767e-04,-2.863360478919873e-04]) 

         # ------------ Initial calibration ---------
         #data = np.loadtxt(params.file_name_calibration) # loads iinvC, invC, ib_0, b_0; [YH] The format I difined here is different from the original
         invC= np.concatenate((np.concatenate((invC, np.zeros((3,3))),axis = 1),np.concatenate((np.zeros((3,3)), np.eye(3)),axis=1)),axis = 0)

         self.inc_msmt= np.transpose(np.dot(iinvC , np.transpose(self.inc_msmt))) - ib_0
         self.msmt= np.transpose(np.dot(invC, np.transpose(self.msmt))) - b_0
         # -------------------------------------------
          
         # ------------ Initial rotation ------------
         # Initial rotation to get: x=foward & z=down
         # load R_init (depends on the horientation of the IMU)
         path = params.path+'IMU/R_init'
         #R_init = np.loadtxt(path)
         R_init = np.array([[0,-1,0],[-1,0,0],[0,0,-1]])
         R_init_block= block_diag(R_init,R_init)
         self.msmt= np.dot(R_init_block ,np.transpose(self.msmt))
         self.inc_msmt= np.dot(R_init, np.transpose(self.inc_msmt))
         # -------------------------------------------
         # reduce the number of epochs to test
         if (params.SWITCH_REDUCE_TESTING ==1):
            self.num_readings= params.num_epochs_static + params.num_epochs_reduce_testing
           
         # ----------------------------------------------
         # ----------------------------------------------
       
     #     function plot_measurements(obj)
      #      % plot accelerometers
     #       figure; hold on; grid on;
     #       plot(obj.time, obj.msmt(1:3,:));
     #         plot(1:length(obj.msmt)', obj.msmt(1:3,:));
     #       legend('acc_x','acc_y','acc_z')
     #       
     #       % plot gyros
     #       figure; hold on; grid on;
     #       plot(obj.time, obj.msmt(4:6,:));
     #       legend('w_x','w_y','w_z')
     #   end
        # ----------------------------------------------
        # ----------------------------------------------



