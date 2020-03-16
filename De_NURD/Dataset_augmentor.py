import cv2
import numpy as np
import os
from analy import MY_ANALYSIS
from analy import Save_signal_enum
from scipy import signal 
import random
from random import seed
from median_filter_special import myfilter
from Correct_sequence_iteration import VIDEO_PEOCESS
from  path_finding import PATH
from scipy.ndimage import gaussian_filter1d

from cost_matrix import COSTMtrix ,Overall_shiftting_WinLen , Window_LEN
class DATA_augmentor(object):
     def __init__(self):
        self.original_root = "../../saved_original_for_generator/"
        self.data_pair1_root = "../../saved_pair1/"
        self.data_pair2_root = "../../saved_pair2/"
        self.data_mat_root = "../../saved_matrix/"
        self.data_mat_root_origin = "../../saved_matrix_unprocessed/"
        self.data_mat_root_augmented = "../../saved_matrix_augmented/"
        self.data_signal_root  = "../../saved_stastics_for_generator/"
        if not os.path.exists(self.data_mat_root_augmented):
            os.mkdir(self.data_mat_root_augmented)
            print("Directory " , self.data_mat_root_augmented ,  " Created ")
        else:    
            print("Directory " , self.data_mat_root_augmented ,  " already exists")
        self.H  = 1024
        self.W = 780
        # read the signals  just use the existing path
        self.saved_stastics = MY_ANALYSIS()
        self.saved_stastics.all_statics_dir = os.path.join(self.data_signal_root, 'signals.pkl')

        self.path_DS =  self.saved_stastics.read_my_signal_results()
        self.path_DS.all_statics_dir  =  self.saved_stastics.all_statics_dir


        #the  validation functionfor check the matrix and can also be used for validate the correction result
     def validation(self,original_IMG,Shifted_IMG,path,Image_ID):
        Costmatrix,shift_used = COSTMtrix.matrix_cal_corre_full_version3_2GPU(original_IMG,Shifted_IMG,0) 
        # Costmatrix  = myfilter.gauss_filter_s(Costmatrix) # smooth matrix
        #tradition way to find path
 
        start_point= PATH.find_the_starting(Costmatrix) # starting point for path searching

        #path_tradition,pathcost1  = PATH.search_a_path(Costmatrix,start_point) # get the path and average cost of the path
        path_deep,path_cost2=PATH.search_a_path_Deep_Mat2longpath(Costmatrix) # get the path and average cost of the path
        path_deep = gaussian_filter1d(path_deep,3) # smooth the path 

        ##middle_point  =  PATH.calculate_ave_mid(mat)
        #path1,path_cost1=PATH.search_a_path(mat,start_point) # get the path and average cost of the path
        show1 =  Costmatrix 
        cv2.imwrite(self.data_mat_root_origin  + str(Image_ID) +".jpg", show1)

        for i in range ( len(path)):
            painter = min(path[i],Window_LEN-1)
            #painter2= min(path_tradition[i],Window_LEN-1)
            painter3 = min(path_deep[i],Window_LEN-1) 
            show1[int(painter),i]=128
            #show1[int(painter2),i]=128
            show1[int(painter3),i]=254

        cv2.imwrite( self.data_mat_root  + str(Image_ID) +".jpg", show1)
     def noisy(self,noise_typ,image):
       if noise_typ == "gauss":
          row,col = image.shape
          mean = 0
          var = 50
          sigma = var**0.5
          gauss = np.random.normal(mean,sigma,(row,col )) 
          gauss = gauss.reshape(row,col ) 
          noisy = image + gauss
          return noisy
       elif noise_typ == "s&p":
          row,col  = image.shape
          s_vs_p = 0.5
          amount = 0.004
          out = np.copy(image)
          # Salt mode
          num_salt = np.ceil(amount * image.size * s_vs_p)
          coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
          out[coords] = 1

          # Pepper mode
          num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
          coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
          out[coords] = 0
          return out
       elif noise_typ == "poisson":
          vals = len(np.unique(image))
          vals = 2 ** np.ceil(np.log2(vals))
          noisy = np.random.poisson(image * vals) / float(vals)
          return noisy
       elif noise_typ =="speckle":
          row,col  = image.shape
          gauss = np.random.randn(row,col )
          gauss = gauss.reshape(row,col )        
          noisy = image + image * gauss
          return noisy

     
     #def add_gaussian_noise(self,X_imgs):
     #       gaussian_noise_imgs = []
     #       row, col, _ = X_imgs[0].shape
     #       # Gaussian distribution parameters
     #       mean = 0
     #       var = 0.1
     #       sigma = var ** 0.5
    
     #       for X_img in X_imgs:
     #           gaussian = np.random.random((row, col, 1)).astype(np.float32)
     #           gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
     #           gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0)
     #           gaussian_noise_imgs.append(gaussian_img)
     #       gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype = np.float32)
     #       return gaussian_noise_imgs

     def augment_gauss_noise(self):
         #read one from the original
        noise_selector=["gauss", "s&p","poisson","speckle"]
        for img in os.listdir(self.data_mat_root_origin):
            a, b = os.path.splitext(img)
            if b == ".jpg":
                original_IMG = cv2.imread(self.data_mat_root_origin+ img)
                original_IMG  =   cv2.cvtColor(original_IMG, cv2.COLOR_BGR2GRAY)
                 
                Gauss_IMG   =  self.noisy(noise_selector[int(a)%4],original_IMG)
            # save all the result
            #cv2.imwrite(self.data_pair1_root  + str(Image_ID) +".jpg", original_IMG)
                cv2.imwrite(self.data_mat_root_augmented  + a +".jpg", Gauss_IMG)
           
         


            print ("[%s]   is processed. test point time is [%f] " % (a ,0.1))

             
     


if __name__ == '__main__':
        generator   = DATA_augmentor()
        generator.augment_gauss_noise ()