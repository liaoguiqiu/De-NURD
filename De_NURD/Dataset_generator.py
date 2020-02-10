import cv2
import numpy as np
import os
from analy import MY_ANALYSIS
from analy import Save_signal_enum
from scipy import signal 
import random
from median_filter_special import myfilter
from Correct_sequence_iteration import VIDEO_PEOCESS
from cost_matrix import COSTMtrix
class DATA_Generator(object):
     def __init__(self):
        self.original_root = "..\\..\\saved_original_for_generator\\"
        self.data_pair1_root = "..\\..\\saved_pair1\\"
        self.data_pair2_root = "..\\..\\saved_pair2\\"
        self.data_mat_root = "..\\..\\saved_matrix\\"
        self.H  = 1024
        self.W = 780
        # read the signals  just use the existing path
        self.saved_stastics = MY_ANALYSIS()
        self.path_DS =  self.saved_stastics.read_my_signal_results()
     def generate(self):
         #read one from the original
            #random select one IMG frome the oringinal 
        read_id = 0
        Len_steam =5
        steam=np.zeros((Len_steam,self.H,self.W)) # create video buffer
        while (1):
            OriginalpathDirlist = os.listdir(self.original_root)    # 
            sample = random.sample(OriginalpathDirlist, 1)  # 
            Sample_path = self.original_root +   sample[0]
            original_IMG = cv2.imread(Sample_path)
            original_IMG  =   cv2.cvtColor(original_IMG, cv2.COLOR_BGR2GRAY)
            original_IMG = cv2.resize(original_IMG, (self.W,self.H), interpolation=cv2.INTER_AREA)

            #read the path and Image number from the signal file
            #get the Id of image which should be poibnt to
            Image_ID = int( self.path_DS.signals[Save_signal_enum.image_iD.value, read_id])
            #get the path
            path  = self.path_DS.path_saving[read_id,:]
            path =  signal.resample(path, self.W)#resample the path
            # create the shifted image
            Shifted_IMG   = VIDEO_PEOCESS.de_distortion(original_IMG,path,Image_ID,0)
            # save all the result
            cv2.imwrite(self.data_pair1_root  + str(Image_ID) +".jpg", original_IMG)
            cv2.imwrite(self.data_pair2_root  + str(Image_ID) +".jpg", Shifted_IMG)
            ## validation 
            steam[Len_steam-1,:,:]  = original_IMG  # un-correct 
            steam[Len_steam-2,:,:]  = Shifted_IMG  # correct 
            Costmatrix,shift_used = COSTMtrix.matrix_cal_corre_full_version3_2GPU(original_IMG,Shifted_IMG,0) 
            Costmatrix  = myfilter.gauss_filter_s (Costmatrix) # smooth matrix
            show1 =  Costmatrix 
            for i in range ( len(path)):
                show1[int(path[i]),i]=254
            cv2.imwrite(self.data_mat_root  + str(Image_ID) +".jpg", show1)



            print ("[%s]   is processed. test point time is [%f] " % (read_id ,0.1))

            read_id +=1


if __name__ == '__main__':
        generator   = DATA_Generator()
        generator.generate()

