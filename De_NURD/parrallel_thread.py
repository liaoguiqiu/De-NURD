from multiprocessing import Process
import numpy as np
from threading import Thread
from cost_matrix import COSTMtrix
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
        return self.costmatrix ,self.shift_used2
 

    # function 1 calculate tthe shift
    def func1(self):
       print('shift star')
       self.overall_shifting,self.shift_used1 = COSTMtrix.Img_fully_shifting_distance (
                                                   self.stream1[self.strmlen-1,:,:],
                                                   self.stream1[0,:,:],  self.add_shift)
       print('shift end')
 
    #calculate the NURD
    def func2(self):
      print('NURD start  ')
      
      self.costmatrix,self.shift_used2= COSTMtrix.matrix_cal_corre_full_version3_2GPU (
                                                              self.stream2[self.strmlen-1,:,:] ,
                                                              self.stream2[self.strmlen-2,:,:], 0) 
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