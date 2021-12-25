from multiprocessing import Process
import numpy as np
from threading import Thread
from cost_matrix import COSTMtrix
from cost_matrix import Standard_LEN,Window_LEN

from shift_deploy import Shift_Predict
from median_filter_special import  myfilter
from path_finding import PATH
from time import time
from pair2path import Pair2Path
import cv2
import matplotlib.pyplot as plt
Graph_searching_flag = False


class Dual_thread_Overall_shift_NURD(object):
    def __init__(self,BranchFlag):
        #give all parmeter initial(given the Memory for thread)
        self.branch_flag = BranchFlag 
        self.stream1 =[]
        self.stream2 =[]
        self.stream3 = []
        self.last_overall_shift  =  0 
        self.overall_shifting =[]
        self.shift_used1 =[]
        self.shift_used2 =[]
        self.costmatrix =[]
        self.strmlen = []
        self.add_shift  = []
        self.shift_predictor = Shift_Predict()
        self.path_predictor  = Pair2Path()
        self.overall_shifting  = 0
        self.shift_used1 =0
        self.costmatrix  = np.zeros((71,832))
        self.shift_used2 =0
        self.path = np.zeros(832)
        # Initiate ORB detector
        self.orb = cv2.ORB_create()
        self.sift = cv2.xfeatures2d.SIFT_create()
        #self.sift = cv2.xfeatures2d.SIFT_create()

        pass
    def input(self,strm1,strm2,strmlen,addshift):
        self.stream1 =strm1
        self.stream2 =strm2
        self.strmlen  = strmlen
        self.add_shift  = addshift
    def inputaddition (self, strm3):
        self.stream3 =  strm3

        pass
    def output_overall(self):
        return self.overall_shifting,self.shift_used1
    def output_NURD(self):
        return self.path,self.costmatrix ,self.shift_used2
 

    # function 1 calculate tthe shift
    def func1(self):
       start_time2 = time()

       print('shift star')
       img1 = self.stream1[self.strmlen-1,:,:]
       self.shift_used1   = self.add_shift

       img2 = self.stream1[self.strmlen-2,:,:]
       img3 = self.stream1[0,:,:]
       img1 = cv2.GaussianBlur(img1,(5,5),0)
       img2 = cv2.GaussianBlur(img2,(5,5),0)
       img3 = cv2.GaussianBlur(img3,(5,5),0)


       H,W = img1.shape 
       #self.overall_shifting2 = self.shift_predictor.predict_shaking(img1,self.stream2[self.strmlen-2,:,:])

       #self.overall_shifting,shift_used1 = COSTMtrix.Img_fully_shifting_correlation (img1[200:H,:],
       #                                                       img3[200:H,:],  self.shift_used1 )
       #self.shift_used1 += self.overall_shifting
       #self.overall_shifting,shift_used1 = COSTMtrix.Img_fully_shifting_correlation (img1[0:210,:],
       #                                                       img3[0:210,:],  self.shift_used1)
       #self.overall_shifting,shift_used1 = COSTMtrix.stack_fully_shifting_correlation (self.stream1[:,0:210,:],
       #                                                       self.stream2[:,0:210,:],  self.shift_used1)
       img1 = np.roll(img1, int(self.shift_used1 ) , axis = 1)     # Positive x rolls right
       crop1 = img1[27:83,:]
       crop3 = img3 [27:83,:]
       crop1=cv2.resize(crop1, (int(W/3) , 30), interpolation=cv2.INTER_LINEAR)
       crop3=cv2.resize(crop3, ( int(W/3),30), interpolation=cv2.INTER_LINEAR)
       crop1=cv2.resize(crop1, (int(W ) , 90), interpolation=cv2.INTER_LINEAR)
       crop3=cv2.resize(crop3, ( int(W ),90), interpolation=cv2.INTER_LINEAR)
       self.overall_shifting,_,matrix= COSTMtrix.Img_fully_shifting_correlation (crop1 ,
                                                              crop3, 0)
            
       #img1_c =  cv2.rotate(img1[30:90,:],rotateCode = cv2.ROTATE_90_CLOCKWISE) 
       #img3_c =  cv2.rotate(img3[30:90,:],rotateCode = cv2.ROTATE_90_CLOCKWISE) 

       ##img3_c = img3[27:200,:]

       #img1c = cv2.cvtColor(img1_c.astype(np.uint8), cv2.COLOR_GRAY2RGB)
       #img3c = cv2.cvtColor(img3_c.astype(np.uint8), cv2.COLOR_GRAY2RGB)
       #img1c=cv2.resize(img1c, (30,int(W/3)), interpolation=cv2.INTER_LINEAR)
       #img3c=cv2.resize(img3c, (30,int(W/3)), interpolation=cv2.INTER_LINEAR)
       #img1c=cv2.resize(img1c, (60,int(W )), interpolation=cv2.INTER_LINEAR)
       #img3c=cv2.resize(img3c, (60,int(W )), interpolation=cv2.INTER_LINEAR)
       ## find the keypoints and descriptors with ORB
       #kp1, des1 = self.sift.detectAndCompute( img1c.astype(np.uint8) ,None)
       #kp2, des2 = self.sift.detectAndCompute( img3c.astype(np.uint8) ,None)
       ## create BFMatcher object
       ##bf = cv2.BFMatcher()
       #bf = cv2.BFMatcher( )

       #matches = bf.knnMatch(des1,des2,k=2)
       # # Sort them in the order of their distance.
       #good = []
       #for m,n in matches:
       #     if m.distance < 0.75*n.distance:
       #         good.append(m)
 

       #res = cv2.drawMatches(img1c,kp1,img3c,kp2,good,None)
       #cv2.imshow("Result", res) 
       #cv2.waitKey(1)
       #plt.imshow(result),plt.show()
       #cv2.wait(1)
       self.shift_used1 += self.overall_shifting
       ###self.overall_shifting = 0 
       img1 = np.roll(img1, int(self.shift_used1 ) , axis = 1)     # Positive x rolls right

       self.overall_shifting = self.shift_predictor.predict(img1,img2,img3) # THIS COST 0.01 s
       self.overall_shifting2 = self.shift_predictor.predict(img2,img1,img1) # THIS COST 0.01 s
       self.overall_shifting = 0.5*self.overall_shifting + 0.5*(Standard_LEN - self.overall_shifting2)
      # self.costmatrix_o,_= COSTMtrix.matrix_cal_corre_block_version3_3GPU  (
      #                                                        img1[0:200,:]  ,
      #                                                        img3 [0:200,:], 0,
      #                                                        block_wid = 3,Down_sample_F = 1,Down_sample_F2 = 2) 

      ##self.costmatrix2,self.shift_used2= COSTMtrix.matrix_cal_corre_block_version3_3GPU  (
      ##                                                        self.stream2[self.strmlen-1,50:211,:] ,
      ##                                                        self.stream2[self.strmlen-2,50:211,:], 0,
      ##                                                        block_wid = 3,Down_sample_F = 5,Down_sample_F2 = 5)
      ###self.costmatrix = self.costmatrix1 
      ##self.costmatrix = 0.6*self.costmatrix1+ 0.4*self.costmatrix2 
      # Hm,Wm= self.costmatrix_o.shape
      # self.costmatrix_o = cv2.resize(self.costmatrix_o, (Wm,Standard_LEN), interpolation=cv2.INTER_AREA)
      # self.path_o  =  PATH.get_warping_vextor(self.costmatrix_o)  # THIS COST 0.03S
      # self.path_o = self.path_o * Window_LEN/Standard_LEN
      # self.overall_shifting = np.mean(self.path_o)
       #self.overall_shifting = 0.7* self.overall_shifting + 0.3*self.overall_shifting2
       #self.overall_shifting =  self.overall_shifting  

       #self.overall_shifting = 0.5*self.overall_shifting + 0.5 * self.last_overall_shift
       #self.last_overall_shift = self.overall_shifting



       #self.overall_shifting,shift_used1 = COSTMtrix.Img_fully_shifting_correlation (img1[0:200,:],
       #                                                       img3[0:200,:],  self.shift_used1 )
                                                  
       print('shift end')
       end_time2 = time()

       print (" B time is [%f] " % ( end_time2 - start_time2))
    #calculate the NURD
    def func2(self):
      start_time = time()
      print('NURD start')
      image1 = self.stream2[self.strmlen-1,:,:]
      h,w = image1.shape
      window_wid = self.path_predictor.Original_window_Len
      self.costmatrix = np.zeros ((window_wid, w))
      
      self.costmatrix,self.shift_used2= COSTMtrix.matrix_cal_corre_block_version3_3GPU  (
                                                              self.stream2[self.strmlen-1,:,:] ,
                                                              self.stream2[self.strmlen-2,:,:], 0,
                                                              block_wid = 3,Down_sample_F = 1,Down_sample_F2 = 2) 

      #self.costmatrix2,self.shift_used2= COSTMtrix.matrix_cal_corre_block_version3_3GPU  (
      #                                                        self.stream2[self.strmlen-1,50:211,:] ,
      #                                                        self.stream2[self.strmlen-2,50:211,:], 0,
      #                                                        block_wid = 3,Down_sample_F = 5,Down_sample_F2 = 5)
      ##self.costmatrix = self.costmatrix1 
      #self.costmatrix = 0.6*self.costmatrix1+ 0.4*self.costmatrix2 
      Hm,Wm= self.costmatrix.shape
      self.costmatrix = cv2.resize(self.costmatrix, (Wm,Standard_LEN), interpolation=cv2.INTER_AREA)

      self.costmatrix  = myfilter.gauss_filter_s (self.costmatrix) # smooth matrix
      #self.costmatrix  = cv2.GaussianBlur(self.costmatrix,(3,3),0)
      #self.costmatrix = self.costmatrix*1.5 +30
        # down sample the materix and up sample 
      #Hm,Wm= self.costmatrix.shape
      #self.costmatrix = cv2.resize(self.costmatrix, (int(Wm/2),int(Hm/2)), interpolation=cv2.INTER_AREA)
      #self.costmatrix = cv2.resize(self.costmatrix, (Wm,Hm), interpolation=cv2.INTER_AREA)


      # THE COST MATRIX COST 0.24 S
      if Graph_searching_flag == True:
          start_point= PATH.find_the_starting(self.costmatrix) # starting point for path searching
          self.path,pathcost1  = PATH.search_a_path(self.costmatrix,start_point)
      else:
          self.path  =  PATH.get_warping_vextor(self.costmatrix)  # THIS COST 0.03S
      self.path = self.path * Window_LEN/Standard_LEN
      #self.path = self.path_predictor.predict(self.stream2[self.strmlen-1,:,:],  self.stream2[self.strmlen-2,:,:])
      end_time = time()
      
      print('NURD end  ')
      print (" A time is [%f] " % ( end_time - start_time))
     
     #return x
    def runInParallel( self):
        if self.branch_flag ==0:
            p1 = Thread(target=self.func1)
            p2 = Thread(target=self.func2)
            p2.start()

            p1.start()
            p2.join()
        
            p1.join()
        elif self.branch_flag == 1:
             
            p2 = Thread(target=self.func2)
            p2.start()

           
            p2.join()
        elif self.branch_flag == 2 :
            p1 = Thread(target=self.func1)
            p1.start()
            p1.join()
    def ruuInCascade(self):
        self.func2()

        self.func1()
 
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