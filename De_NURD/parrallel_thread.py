from multiprocessing import Process
import numpy as np
from threading import Thread
from cost_matrix import COSTMtrix
from shift_deploy import Shift_Predict
from median_filter_special import  myfilter
from path_finding import PATH
from time import time
import cv2
class Dual_thread_Overall_shift_NURD(object):
    def __init__(self):
        #give all parmeter initial(given the Memory for thread)
        self.stream1 =[]
        self.stream2 =[]
        self.overall_shifting =[]
        self.shift_used1 =[]
        self.shift_used2 =[]
        self.costmatrix =[]
        self.strmlen = []
        self.add_shift  = []
        self.shift_predictor = Shift_Predict()
        pass
    def input(self,strm1,strm2,strmlen,addshift):
        self.stream1 =strm1
        self.stream2 =strm2
        self.strmlen  = strmlen
        self.add_shift  = addshift

        pass
    def output_overall(self):
        return self.overall_shifting,self.shift_used1
    def output_NURD(self):
        return self.path,self.costmatrix ,self.shift_used2
 

    # function 1 calculate tthe shift
    def func1(self):
       print('shift star')
       img1 = self.stream1[self.strmlen-1,:,:]
       self.shift_used1   = self.add_shift

       img2 = self.stream1[self.strmlen-2,:,:]
       img3 = self.stream1[0,:,:]
       H,W = img1.shape 
       #self.overall_shifting,shift_used1 = COSTMtrix.Img_fully_shifting_correlation (img1[200:H,:],
       #                                                       img3[200:H,:],  self.shift_used1 )
       #self.shift_used1 += self.overall_shifting
       self.overall_shifting,shift_used1 = COSTMtrix.Img_fully_shifting_correlation (img1[0:210,:],
                                                              img3[0:210,:],  self.shift_used1 )
       self.shift_used1 += self.overall_shifting
       img1 = np.roll(img1, self.shift_used1  , axis = 1)     # Positive x rolls right
       self.overall_shifting = self.shift_predictor.predict(img1,img2,img3) # THIS COST 0.01 s

       #self.overall_shifting,shift_used1 = COSTMtrix.Img_fully_shifting_correlation (img1[0:200,:],
       #                                                       img3[0:200,:],  self.shift_used1 )
                                                  
       print('shift end')
 
    #calculate the NURD
    def func2(self):
      print('NURD start  ')

      
      self.costmatrix,self.shift_used2= COSTMtrix.matrix_cal_corre_full_version3_2GPU  (
                                                              self.stream2[self.strmlen-1,:,:] ,
                                                              self.stream2[self.strmlen-2,:,:], 0) 
      #self.costmatrix,self.shift_used2= COSTMtrix.     matrix_cal_corre_full_version_2  (self.stream2,0)
      #self.costmatrix  = myfilter.gauss_filter_s (self.costmatrix) # smooth matrix
      #self.costmatrix  = cv2.GaussianBlur(self.costmatrix,(5,5),0)
        # down sample the materix and up sample 
      #Hm,Wm= self.costmatrix.shape
      #self.costmatrix = cv2.resize(self.costmatrix, (int(Wm/2),int(Hm/2)), interpolation=cv2.INTER_AREA)
      #self.costmatrix = cv2.resize(self.costmatrix, (Wm,Hm), interpolation=cv2.INTER_AREA)


      # THE COST MATRIX COST 0.24 S

      self.path  =  PATH.get_warping_vextor(self.costmatrix)  # THIS COST 0.03S
      #start_point= PATH.find_the_starting(self.costmatrix) # starting point for path searching
      #self.path,pathcost1  = PATH.search_a_path(self.costmatrix,start_point)
      print('NURD end  ')
     
     #return x
    def runInParallel( self):
 
        p1 = Thread(target=self.func1)
        p2 = Thread(target=self.func2)

        p1.start()
        p2.start()
        
        p1.join()
        p2.join()

 
if __name__ == '__main__':
    operator = Dual_thread_Overall_shift_NURD()
 
    operator. runInParallel()
    #p1 = Process(target=operator.func1)
    #p1.start()
    #p2 = Process(target=operator.func2)
    #p2.start()
    #thread1 = Thread(target = operator.func1) 
    #thread2 = Thread(target = operator.func2) 
    #thread1.start()
    #thread2.start()

    #thread1.join()
    #thread2.join()

    print(str(operator.costmatrix  ) +'final')
    x=1
    x=1