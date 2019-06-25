import numpy as np
import math



def  pi_to_pi(angle):
     if angle.shape[0] == 1:
        if (angle < -2*math.pi or angle > 2*math.pi):
           #warning('pi_to_pi() error: angle outside 2-PI bounds.')
           angle= np.mod(angle, 2*pi)

        if angle > pi:
          angle= angle - 2*math.pi
        elif angle < -pi:
             angle= angle + 2*math.pi


     else:
         acc = 0
         tmp = []
         for i in angle:
             if(i<-2*math.pi or i>2*math.pi):
               tmp.append(acc)     #i= find(angle<-2*pi | angle>2*pi) # replace with a check
             acc = acc+1
         if(len(tmp)!=0): 
            #    warning('pi_to_pi() error: angle outside 2-PI bounds.')
            for j in angle: #angle(i) = mod(angle(i), 2*pi)
                angle[acc] = np.mod(angle[j],2*math.pi)
     acc = 0
     tmp = []
     tmp2 = []
     for i in angle:
         if(i>math.pi):
           tmp.append(acc)
         if(i<-math.pi):
            tmp2.append(acc)
         acc = acc+1
     for i in tmp:
         angle[i] = angle[i] - 2*math.pi
         angle[i] = angle[i] + 2*math.pi

     return angle

