
import cv2
import numpy as np
import os
from analy import MY_ANALYSIS
from analy import Save_signal_enum
from scipy import signal 
import scipy.io
import random
from random import seed
from median_filter_special import myfilter
from Correct_sequence_iteration import VIDEO_PEOCESS
from  path_finding import PATH
from scipy.ndimage import gaussian_filter1d
import pickle
from shift_deploy import Shift_Predict
from cost_matrix import COSTMtrix ,Overall_shiftting_WinLen , Window_LEN
from Dataset_generator import DATA_Generator
class Communicate(object):
    def __init__(self ):
        #set = Read_read_check_ROI_label()
        #self.database_root = set.database_root
        #check or create this path
        #self.self_check_path_create(self.signal_data_path)
        self.training= 1
        self.writing = 2
        self.pending = 1
    def change_state(self):
        if self.writing ==1:
           self.writing =0
        pass
    def read_data(self,dir):
        saved_path  = dir  + 'protocol.pkl'
        self = pickle.load(open(saved_path,'rb'),encoding='iso-8859-1')
        return self
    def save_data(self,dir):
        #save the data 
        save_path = dir + 'protocol.pkl'
        with open(save_path , 'wb') as f:
            pickle.dump(self , f, pickle.HIGHEST_PROTOCOL)
        pass




if __name__ == '__main__':
    generator  = DATA_Generator()
     
    talker = Communicate()
    com_dir = "../../../../../" + "Deep learning/dataset/telecom/deeppath/"

    talker=talker.read_data(com_dir)
    #initialize the protocol
    #talker.pending = 0
    #talker=talker.save_data(com_dir)

    #generator.save_img_dir = "../../../../../"  + "Deep learning/dataset/"
    #generator.save_contour_dir = "../../"     + "saved_stastics_coutour_generated/"

    imgbase_dir = "../../../../../"  + "Deep learning/dataset/CostMatrix/"
    labelbase_dir = "../../../../../"  + "Deep learning/dataset/saved_stastics/"

    while(1):
        generator  = DATA_Generator()


        talker=talker.read_data(com_dir)

        if talker.training==1 and talker.writing==2: # check if 2 need writing
            if talker.pending == 0 :
                #generator.data_mat_root_origin = "../../saved_matrix_unprocessed/"
                generator.data_mat_root_origin = imgbase_dir+"2/"
                generator.path_DS.all_statics_dir  = labelbase_dir+"2/"
                generator.path_DS.all_statics_dir=os.path.join(generator.path_DS.all_statics_dir, 'signals.pkl')

                #generator.save_img_dir = imgbase_dir+"2/"
                #generator.save_contour_dir =  labelbase_dir+"2/"

                generator.generate_NURD() # generate

                talker.writing=1
                talker.pending=1
                talker.save_data(com_dir)
        if talker.training==2 and talker.writing==1: # check if 2 need writing
            if talker.pending == 0 :
                generator.data_mat_root_origin = imgbase_dir+"1/"
                generator.path_DS.all_statics_dir =  labelbase_dir+"1/"
                generator.path_DS.all_statics_dir=os.path.join(generator.path_DS.all_statics_dir, 'signals.pkl')

                generator.generate_NURD() # generate

                talker.writing=2
                talker.pending=1
                talker.save_data(com_dir)
        cv2.waitKey(1000)   
        print("waiting")



    
 