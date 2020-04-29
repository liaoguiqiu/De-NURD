import pickle
from scipy import signal 
import scipy.io
import numpy as np
import os

########################class for signal##########################################
class Save_Signal_matlab(object):
      def __init__(self):
          self.flag  = True

          self.save_matlab_root = "../../saved_matlab/"
          self.check_dir(self.save_matlab_root)

          self.infor_shift_nurd_dir = "../../saved_matlab/infor_shift_NURD.mat"
          self.label = []
          self.truth = []
          self.deep_result =[]
          self.tradition_result = []
          self.overall_shift = []
          self.NURD   = []
          self.loss  = []
          pass
      def check_dir(self,this_path):
        if not os.path.exists(this_path):
            os.mkdir(this_path)
            print("Directory " , this_path ,  " Created ")
        else:    
            print("Directory " , this_path ,  " already exists")
      def buffer_loss(self,input):
            self.loss.append(input)
         
            pass 
      def buffer_4(self,id,truth,deep,tradition):
          self.label.append(id)
          self.truth.append(truth)
          self.deep_result.append(deep)
          self.tradition_result.append(tradition)
          pass
      def buffer_overall_shift_NURD(self,id,overall_shift, nurd):
            self.label.append(id)
            self.overall_shift.append(overall_shift)
            self.NURD.append(nurd)
            pass
      def save_mat(self):
          scipy.io.savemat(self.save_matlab_root+'result.mat', mdict={'arr': self})
          pass


      def save_mat_infor_of_over_allshift_with_NURD(self):
            scipy.io.savemat(self.save_matlab_root+'infor_shift_NURD.mat', mdict={'arr': self})
            pass
      def save_pkl_infor_of_over_allshift_with_NURD(self):
          with open(self.save_matlab_root+'infor_shift_NURD.pkl', 'wb') as f:
            pickle.dump(self , f, pickle.HIGHEST_PROTOCOL)
      def read_pkl_infor_of_over_allshift_with_NURD(self):
          result =     pickle.load(open(self.save_matlab_root+'infor_shift_NURD.pkl','rb'),encoding='iso-8859-1')
          #decode the mat data 
          nurd = result.NURD
          shift = result.overall_shift
          id = result.label
          return id,nurd,shift
#####################class for generat function#############################################
     
