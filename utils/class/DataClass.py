import PredictionClass
import UpdateDataClass
import IntegrityDataClass
import numpy as np
import matplotlib as mpl
from mpl_toolkits import mplot3d
import matplotlib.pylab as plt
class DataClass:

      pred = None
      update = None
      im = None
      num_extracted_features = None
      msmts = None        
      gps_msmts = None
    
    
      # ----------------------------------------------
      # ----------------------------------------------
      def __init__(self,imu_num_readings, lidar_num_readings, params):
          self.pred= PredictionClass.PredictionDataClass(imu_num_readings, params)
          self.update= UpdateDataClass.UpdateDataClass(imu_num_readings, params)
          self.im= IntegrityDataClass.IntegrityDataClass(lidar_num_readings)
        
      # ----------------------------------------------
      # ----------------------------------------------
      def store_prediction(self, epoch, estimator, time):
          self.pred.store(epoch, estimator, time)
        
      # ----------------------------------------------
      # ----------------------------------------------
      def store_prediction_sim(self, epoch, estimator, time):
          self.pred.store_sim(epoch, estimator, time)
        
      # ----------------------------------------------
      # ----------------------------------------------
      def store_update(self, epoch, estimator, time):
          self.update.store(epoch, estimator, time)
            
          # increase counter
          epoch= epoch + 1
          return epoch
        
      # ----------------------------------------------
      # ----------------------------------------------
      def store_update_fg(self, epoch, estimator, time, params):
          self.update.store_fg(epoch, estimator, time, params)
            
          # increase counter
          epoch= epoch + 1
         
          return epoch

      def  store_update_sim(self, epoch, estimator, time, params):
          self.update.store_sim(epoch, estimator, time, params)
            
            # increase counter
          epoch= epoch + 1
          return epoch
      # ----------------------------------------------
      # ----------------------------------------------
      def store_msmts(self, msmts): # TODO: optimize this mess!!
          self.num_extracted_features= np.array([[self.num_extracted_features],[msmts.shape[1]]])
          if(self.msmts is None ):
             self.msmts = msmts
          else:   
             self.msmts= np.concatenate((self.msmts,msmts),axis = 0)
        
      # ----------------------------------------------
      # ----------------------------------------------
      def store_gps_msmts(self, gps_msmt): # TODO: optimize this mess!!
          self.gps_msmts= np.array([self.gps_msmts,np.transpose(gps_msmt)])
        
      # ----------------------------------------------
      # ----------------------------------------------
      def store_integrity_data(self, im, estimator, counters, params):
          self.im.store(im, estimator, counters, params)
        
      # ----------------------------------------------
      # ----------------------------------------------    
      def delete_extra_allocated_memory(self, counters):
          self.update.delete_extra_allocated_memory(counters)
          self.im.delete_extra_allocated_memory(counters)
        
      # ----------------------------------------------
      # ----------------------------------------------
      def find_HMI_sim(self, params):
      # this def finds the indexes where HMI occurs, i.e. where
      # the error surpasses the alert limit and the detector is not
      # triggered
            
          HMI_inds= None
          fail_ind = []
          acc = 0
          #fail_ind= find( self.update.error_state_interest > params.alert_limit )
          for i in self.update.error_state_interest:
              if(i>params.alert_limit):
                 fail_ind.append(acc)
              acc = acc+1
            
          if (not fail_ind): 
             return 0
            
          for i in range(len(fail_ind)):
              if (self.update.q_d[fail_ind[i]] < self.update.T_d[fail_ind[i]]):
                  HMI_inds= np.array([[HMI_inds],[fail_ind[i]]])

          return HMI_inds
                
      def plot_map_slam(self, estimator, gps, num_readings, params):
      # Plot GPS+IMU estimated path

          fig = plt.figure()
          ax = fig.add_subplot(111, projection='3d')

          ax.scatter((self.pred.XX[0,:]), (self.pred.XX[1,:]), (self.pred.XX[2,:]), c='r', marker='o',s=0.5)
          ax.scatter((gps.msmt[0,:]),(gps.msmt[1,:]),(gps.msmt[2,:]),c='b',marker = '*',s = 0.5)
          ax.scatter((self.update.XX[0,:]),(self.update.XX[1,:]),(self.update.XX[2,:]),c='y',marker = '*',s = 0.5)
          if (estimator.num_landmarks>0):
              lm_map= np.concatenate((estimator.XX[15::2], estimator.XX[16::2]),axis = 1)
              lm_to_eleminate = []
              for i in range(1,estimator.num_landmarks+1):
                  if (estimator.appearances[i-1]<params.min_appearances):
                     lm_to_eleminate.append(i)
          ax.scatter((lm_map[:,0]), (lm_map[:,1]),(np.zeros(lm_map.shape[0])),c = 'g',marker = 'x',s = 1.0)  
          #ax.scatter((self.msmts[:,0]), (self.msmts[:,1]),(np.zeros(lm_map.shape[0])),c = 'b',marker = 'x',s = 1.0)   
              

          ax.set_xlabel('X Label')
          ax.set_ylabel('Y Label')
          ax.set_zlabel('Z Label')
          ax.set_zlim(-90, 70)
          plt.show()
      '''      
        # ----------------------------------------------
        # ----------------------------------------------
        lm_map= plot_map_slam(self, estimator, gps, num_readings, params)
        # ----------------------------------------------
        # ----------------------------------------------
        plot_map_localization(self, estimator, gps, num_readings, params)
        # ----------------------------------------------
        # ----------------------------------------------
        plot_map_localization_sim(self, estimator, num_readings, params)
        # ----------------------------------------------
        # ----------------------------------------------
        plot_map_localization_fg(self, estimator, params)
        # ----------------------------------------------
        # ----------------------------------------------
        def plot_detector(self, params)
            # plot the detector value Vs. the detector threshold to see if 
            # an alarm has been triggered
            
            figure hold on grid on
            if params.SWITCH_SIM
                plot(self.im.time * params.velocity_sim, self.im.detector, 'linewidth', 2)
                plot(self.im.time * params.velocity_sim, self.im.detector_threshold, 'linewidth', 2)
                xlabel('x [m]')
            else
                plot(self.im.time, self.im.detector, 'linewidth', 2)
                plot(self.im.time, self.im.detector_threshold, 'linewidth', 2)
                xlabel('time [s]')
            
            leg('detector', 'threshold')
        
        # ----------------------------------------------
        # ----------------------------------------------
        def plot_detector_fg(self, params)
            figure hold on grid on
            plot(self.update.time, self.update.q_d, 'linewidth', 2)
            plot(self.update.time, self.update.T_d, 'linewidth', 2)
            leg('detector', 'threshold')
        
        # ----------------------------------------------
        # ----------------------------------------------
        def plot_MA_probabilities(self)
            figure hold on grid on
            for i= 1:length(self.im.time)
                # if it's empty --> continue
                if isempty(self.im.association), continue, 
                
                # take the landmark indexes
                lm_inds= self.im.association{i}
                P_MA= self.im.P_MA_k{i}
                
                # plot
                for j= 1:length(lm_inds)
                    plot( lm_inds(j), P_MA(j), 'bo' )
                
            
            xlabel('landmark ID')
            ylabel('P(MA)')
        
        # ----------------------------------------------
        # ----------------------------------------------
        def plot_P_H(self)
            figure hold on grid on
            for i= 1:length(self.im.time)
                time= self.im.time(i)
                P_H= self.im.P_H{i}
                if isempty(P_H), continue, 
                
                for j= 1:length(P_H)
                    plot(time, P_H(j), '.')
                
            
            xlabel('time [s]')
            ylabel('P(H)')
        
        # ----------------------------------------------
        # ----------------------------------------------
        def plot_error(self, params)
#             standard_dev_x= sqrt( self.update.PX(1,:) )
            standard_dev_y= sqrt( self.update.PX(2,:) )
            
            figure hold on grid on
#             plot(self.update.time * params.velocity_sim, self.update.error(1,:), 'b-', 'linewidth', 2)
            plot(self.update.time * params.velocity_sim, self.update.error(2,:), 'r-', 'linewidth', 2)
#             plot(self.update.time * params.velocity_sim, standard_dev_x,'b--','linewidth',2)
#             plot(self.update.time * params.velocity_sim, -standard_dev_x,'b--','linewidth',2)
            plot(self.update.time * params.velocity_sim, standard_dev_y,'r--','linewidth',2)
            plot(self.update.time * params.velocity_sim, -standard_dev_y,'r--','linewidth',2)
            
            leg({'$\delta \hat{x}$', '$\hat{\sigma}$'},'interpreter', 'latex','fontsize', 15)
            xlabel('x [m]','interpreter', 'latex','fontsize', 15)
            ylabel('error [m]','interpreter', 'latex','fontsize', 15)
        
        # ----------------------------------------------
        # ----------------------------------------------
        def plot_error_fg(self, params)
            
            figure hold on grid on

            plot(self.update.time,  self.update.error_state_interest(:), 'r-', 'linewidth', 2)
#             plot(self.update.time,  self.update.sig_state_interest(:) ,'r--','linewidth',2)
#             plot(self.update.time, -self.update.sig_state_interest(:),'r--','linewidth',2)
            
#             leg({'$\delta \hat{x}$', '$\hat{\sigma}$'},'interpreter', 'latex','fontsize', 15)
            xlabel('time [s]','interpreter', 'latex','fontsize', 15)
            ylabel('error [m]','interpreter', 'latex','fontsize', 15)
        
        # ----------------------------------------------
        # ----------------------------------------------
        def plot_estimates(self)
            
            # Plot variance estimates
            standard_dev= sqrt( self.update.PX )
            
            # Plot SD -- pose
            figure hold on title('Standard Deviations')
            
            subplot(3,3,1) hold on grid on
            plot(self.update.time, standard_dev(1,:),'b-','linewidth',2)
            ylabel('x [m]')
            
            subplot(3,3,2) hold on grid on
            plot(self.update.time, standard_dev(2,:),'r-','linewidth',2)
            ylabel('y [m]')
            
            subplot(3,3,3) hold on grid on
            plot(self.update.time, standard_dev(3,:),'g-','linewidth',2)
            ylabel('z [m]')
            
            subplot(3,3,4) hold on grid on
            plot(self.update.time, standard_dev(4,:),'b-','linewidth',2)
            ylabel('v_x [m/s]')
            
            subplot(3,3,5) hold on grid on
            plot(self.update.time, standard_dev(5,:),'r-','linewidth',2)
            ylabel('v_y [m/s]')
            
            subplot(3,3,6) hold on grid on
            plot(self.update.time, standard_dev(6,:),'g-','linewidth',2)
            ylabel('v_z [m/s]')
            
            subplot(3,3,7) hold on grid on
            plot(self.update.time, rad2deg(standard_dev(7,:)),'b-','linewidth',2)
            ylabel('\phi [deg]') xlabel('Time [s]')
            
            subplot(3,3,8) hold on grid on
            plot(self.update.time, rad2deg(standard_dev(8,:)),'r-','linewidth',2)
            ylabel('\theta [deg]') xlabel('Time [s]')
            
            subplot(3,3,9) hold on grid on
            plot(self.update.time, rad2deg(standard_dev(9,:)),'g-','linewidth',2)
            ylabel('\psi [deg]') xlabel('Time [s]')
        
        # ----------------------------------------------
        # ----------------------------------------------
        def plot_integrity_risk(self, params)
            figure hold on grid on
            if params.SWITCH_SIM
                plot(self.im.time * params.velocity_sim, self.im.p_hmi, 'b-', 'linewidth', 2)
                xlabel('x [m]','interpreter', 'latex','fontsize', 15)
                xlim([self.im.time(1), self.im.time()] * params.velocity_sim) # reset the x-axis (otherwise it moves)
            else
                if params.SWITCH_FACTOR_GRAPHS
                    plot(self.im.time, self.im.p_hmi, 'b-', 'linewidth', 2)
                    xlabel('Time [s]','interpreter', 'latex','fontsize', 15)
                    xlim([self.im.time(1), self.im.time()]) # reset the x-axis (otherwise it moves)
                else
                    plot(self.im.time, self.im.p_hmi, 'b-', 'linewidth', 2)
                    xlabel('Time [s]','interpreter', 'latex','fontsize', 15)
                    xlim([self.im.time(1), self.im.time()]) # reset the x-axis (otherwise it moves)
                
            
            # plot(self.im.time, self.im.p_eps, 'r-', 'linewidth', 2)
            ylabel('P(HMI)','interpreter', 'latex','fontsize', 15)
            set(gca, 'YScale', 'log')
            ylim([1e-15,1])
        
        # ----------------------------------------------
        # ----------------------------------------------
        def plot_number_of_landmarks(self, params)
            figure hold on grid on
            if params.SWITCH_SIM
                plot(self.im.time * params.velocity_sim, self.im.n_L_M, 'b-', 'linewidth', 2)
                plot(self.update.time * params.velocity_sim, self.update.num_associated_lms, 'g-', 'linewidth', 2)
#                 plot(self.im.time * params.velocity_sim, self.update.miss_associations, 'r*')
#                 plot(self.update.time * params.velocity_sim, self.update.num_of_extracted_features, 'k-', 'linewidth', 2)
                xlabel({'x [m]'},'interpreter', 'latex','fontsize', 15)
                leg({'$n^{F^(M)}$', '$n^F$'},...
                    'interpreter', 'latex','fontsize', 15)
            else
                plot(self.im.time, self.im.n_L_M, 'b-', 'linewidth', 2)
                plot(self.update.time, self.update.num_associated_lms, 'g-', 'linewidth', 2)
#                 plot(self.update.time, self.update.num_of_extracted_features, 'k-', 'linewidth', 2)
                xlabel('time [s]','interpreter', 'latex')
                leg({'$n^{F^(M)}$', '$n^F$'},'interpreter', 'latex','fontsize', 15)
            
#             ylabel('Number of landmarks','interpreter', 'latex','fontsize', 15)
        
        # ----------------------------------------------
        # ----------------------------------------------
        def plot_number_of_landmarks_fg_sim(self, params)
            figure hold on grid on
            plot(self.update.time, self.update.n_L_M, 'b-', 'linewidth', 2)
            plot(self.update.time, self.update.n_L_k, 'g-', 'linewidth', 2)
            if ~params.SWITCH_OFFLINE
                plot(self.update.time, self.update.num_faults, 'r-', 'linewidth', 2)
            
            xlabel({'x [m]'},'interpreter', 'latex','fontsize', 15)
            leg({'$n^{L^(M)}$', '$n^L$', '$n_{f}$'},...
                'interpreter', 'latex','fontsize', 15)
        
        # ----------------------------------------------
        # ----------------------------------------------
        def plot_number_epochs_in_preceding_horizon(self, params)
            figure hold on grid on
            if params.SWITCH_SIM
                plot(self.im.time * params.velocity_sim, self.im.M, 'b-', 'linewidth', 2)
                xlabel('x [m]','interpreter', 'latex','fontsize', 15)
            else
                plot(self.im.time, self.im.M, 'b-', 'linewidth', 2)
                xlabel('time [s]','interpreter', 'latex','fontsize', 15)
            
            ylabel('M', 'interpreter', 'latex','fontsize', 15)
        
        # ----------------------------------------------
        # ----------------------------------------------
        def plot_bias_calibration(self)
            figure hold on grid on
            plot(self.update.time, self.update.XX(10:12,:), 'linewidth',2)
            leg({'$acc_x$','$acc_y$','$acc_z$'},'interpreter', 'latex','fontsize', 15)
            
            figure hold on grid on
            plot(self.update.time, self.update.XX(13:15,:), 'linewidth',2)
            leg({'$w_x$','$w_y$','$w_z$'},'interpreter', 'latex','fontsize', 15)
        
        # ----------------------------------------------
        # ----------------------------------------------
       '''

