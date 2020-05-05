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
visdom_show_flag = False
Save_matlab_flag = True
if visdom_show_flag == True:
    from analy_visdom import VisdomLinePlotter

add_noise_flag  = False
Clip_matrix_flag = True
NURD_remove_shift_flag= True 

# 
Show_circular_flag   = True
########################class for signal##########################################
class Save_Signal_matlab(object):
      def __init__(self):
          self.save_matlab_root = "../../saved_matlab/"

          self.infor_shift_nurd_dir = "../../saved_matlab/infor_shift_NURD.mat"
          self.label = []
          self.truth = []
          self.deep_result =[]
          self.tradition_result = []
          self.overall_shift = []
          self.NURD   = []
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
          scipy.io.savemat(self.save_matlab_root+'validation_between_frame.mat', mdict={'arr': self})
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
         
class DATA_Generator(object):
     
     def __init__(self):
        self.original_root = "../../saved_original_for_generator/"
        self.data_pair1_root = "../../saved_pair1/"
        self.data_pair2_root = "../../saved_pair2/"
        self.data_mat_root = "../../saved_matrix/"
        self.data_mat_root_origin = "../../saved_matrix_unprocessed/"
        self.data_signal_root  = "../../saved_stastics_for_generator/"
        self.noise_selector=['gauss_noise','gauss_noise','gauss_noise','gauss_noise']
        self.save_matlab_root = "../../saved_matlab/"
        self.self_check_path_create(self.save_matlab_root  )
        self.H  = 1024
        self.W = 832
        self.matlab = Save_Signal_matlab()
        # read the signals  just use the existing path
        self.saved_stastics = MY_ANALYSIS()
        self.saved_stastics.all_statics_dir = os.path.join(self.data_signal_root, 'signals.pkl')
        self.shift_predictor  = Shift_Predict()
        self.path_DS =  self.saved_stastics.read_my_signal_results()
        self.path_DS.all_statics_dir  =  self.saved_stastics.all_statics_dir
        
        if visdom_show_flag == True:
            self.vis_ploter = VisdomLinePlotter()
       # if there is no path, generate thsi path
     def self_check_path_create(self,directory):
            try:
                os.stat(directory)
            except:
                os.mkdir(directory)   
  # read the original signal and then close the startign with the end
     def close_the_origin_signal(self):
         read_id = 0   # read pointer initialization
         while (1):
             
            OriginalpathDirlist = os.listdir(self.original_root)    # 
             # 
 

            #read the path and Image number from the signal file
            #get the Id of image which should be poibnt to
 
            #get the path
            path  = self.path_DS.path_saving[read_id,:]
            path_l  = len(path)
            long_path = np.append(path,path,axis=0)
            long_path = np.append(long_path,path,axis=0)
            #long_path = np.append(path[::-1],path,axis=0)
            #long_path = np.append(long_path,path[::-1],axis=0)
            long_path= gaussian_filter1d(long_path,5) # the fileter parameters is 5
            path_p = long_path[path_l:2*path_l]


            #change the signal too
            self.path_DS.path_saving[read_id,:] = path_p
 
 
            self.path_DS.save()
 

            ## validation 
            #steam[Len_steam-1,:,:]  = original_IMG  # un-correct 
            #steam[Len_steam-2,:,:]  = Shifted_IMG  # correct 
            #Costmatrix,shift_used = COSTMtrix.matrix_cal_corre_full_version3_2GPU(original_IMG,Shifted_IMG,0) 
            ##Costmatrix  = myfilter.gauss_filter_s (Costmatrix) # smooth matrix
            #show1 =  Costmatrix 
            #for i in range ( len(path)):
            #    show1[int(path[i]),i]=254
            #cv2.imwrite(self.data_mat_root  + str(Image_ID) +".jpg", show1)



            print ("[%s]   is processed. test point time is [%f] " % (read_id ,0.1))

            read_id +=1
         pass
     def add_lines_to_matrix(self,matrix):
        value  = 128
        H,W = matrix.shape
        line_positions = np.arange(0,W-2*H,H)
        for lines in line_positions:
            for i  in np.arange (0, H):
                matrix[i,lines+i] =value
                matrix[i,lines+i+1] =value
                matrix[i,lines+i+3] =value

        return matrix  
     def random_min_clip_by_row(self,min1,min2,mat):
         rand= np.random.random_sample()
         rand = rand * (min2-min1) +min1
         H,W = mat.shape
         for i in np.arange(W):
             rand= np.random.random_sample()
             rand = rand * (min2-min1) +min1
             mat[:,i] = np.clip(mat[:,i],rand,254)
         return mat
     def noisy(self,noise_typ,image):
           if noise_typ == "gauss_noise":
              row,col = image.shape
              mean = 0
              var = 50
              sigma = var**0.5
              gauss = np.random.normal(mean,sigma,(row,col )) 
              gauss = gauss.reshape(row,col ) 
              noisy = image + gauss
              return np.clip(noisy,0,254)
           elif noise_typ == 's&p':
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
              return np.clip(out,0,254)
           elif noise_typ == 'poisson':
              vals = len(np.unique(image))
              vals = 2 ** np.ceil(np.log2(vals))
              noisy = np.random.poisson(image * vals) / float(vals)
              return np.clip(noisy,0,254)
           elif noise_typ =='speckle':
              row,col  = image.shape
              gauss = np.random.randn(row,col )
              gauss = gauss.reshape(row,col )        
              noisy = image + image * gauss
              return np.clip(noisy,0,254)

        #the  validation functionfor check the matrix and can also be used for validate the correction result
     def validation_shift(self,original_IMG,Shifted_IMG,path,Image_ID):
        Costmatrix,shift_used = COSTMtrix.matrix_cal_corre_full_version3_2GPU(original_IMG,Shifted_IMG,0) 
        if Clip_matrix_flag == True:
            #Costmatrix = np.clip(Costmatrix, 20,254)
            Costmatrix=self.random_min_clip_by_row(15,30,Costmatrix)
        Shifted_IMG2 = Shifted_IMG
        shift  = self.shift_predictor.predict(original_IMG,Shifted_IMG,Shifted_IMG2)
        path_deep = shift + path*0

        ##middle_point  =  PATH.calculate_ave_mid(mat)
        #path1,path_cost1=PATH.search_a_path(mat,start_point) # get the path and average cost of the path
        show1 = np.zeros((Costmatrix.shape[0] , Costmatrix.shape[1],3))   
        cv2.imwrite(self.data_mat_root_origin  + str(Image_ID) +".jpg", show1)
        show1[:,:,0] = Costmatrix
        show1[:,:,1] = Costmatrix
        show1[:,:,2] = Costmatrix

        for i in range ( len(path)):
            painter = min(path[i],Window_LEN-1)
            #painter2= min(path_tradition[i],Window_LEN-1)
            painter3 = min(path_deep[i],Window_LEN-1) 
            show1[int(painter),i,:]=[255,255,255]
            #show1[int(painter2),i,:]=[254,0,0]
            show1[int(painter3),i,:]=[0,0,254]
        # save the  matrix to fil dir
        cv2.imwrite( self.data_mat_root  + str(Image_ID) +".jpg", show1)

     def validation(self,original_IMG,Shifted_IMG,path,Image_ID):
        #Costmatrix,shift_used = COSTMtrix.matrix_cal_corre_full_version3_2GPU(original_IMG,Shifted_IMG,0) 
        Costmatrix,shift_used = COSTMtrix.matrix_cal_corre_full_version3_2GPU(original_IMG,Shifted_IMG,0) 
        #Costmatrix=cv2.blur(Costmatrix,(2,2))
        Costmatrix  = myfilter.gauss_filter_s (Costmatrix) # smooth matrix
        if Clip_matrix_flag == True:
            #Costmatrix = np.clip(Costmatrix, 20,254)
            Costmatrix=self.random_min_clip_by_row(15,30,Costmatrix)
        #Costmatrix = self.add_lines_to_matrix(Costmatrix)
        #Costmatrix=np.clip(Costmatrix, 20, 255)
        # Costmatrix  = myfilter.gauss_filter_s(Costmatrix) # smooth matrix
        #tradition way to find path
 
        start_point= PATH.find_the_starting(Costmatrix) # starting point for path searching

        path_tradition,pathcost1  = PATH.search_a_path(Costmatrix,start_point) # get the path and average cost of the path
        #path_deep,path_cost2=PATH.search_a_path_Deep_Mat2longpath(Costmatrix) # get the path and average cost of the path
        path_deep,path_cost2=PATH.search_a_path_deep_multiscal_small_window_fusion(Costmatrix) # get the path and average cost of the path
        
        path_deep = gaussian_filter1d(path_deep,2) # smooth the path 

        ##middle_point  =  PATH.calculate_ave_mid(mat)
        #path1,path_cost1=PATH.search_a_path(mat,start_point) # get the path and average cost of the path
        show1 = np.zeros((Costmatrix.shape[0] , Costmatrix.shape[1],3))   
        cv2.imwrite(self.data_mat_root_origin  + str(Image_ID) +".jpg", show1)
        show1[:,:,0] = Costmatrix
        show1[:,:,1] = Costmatrix
        show1[:,:,2] = Costmatrix

        for i in range ( len(path)):
            painter = min(path[i],Window_LEN-1)
            painter2= min(path_tradition[i],Window_LEN-1)
            painter3 = min(path_deep[i],Window_LEN-1) 
            show1[int(painter),i,:]=[255,255,255]
            show1[int(painter2),i,:]=[254,0,0]
            show1[int(painter3),i,:]=[0,0,254]
        # save the  matrix to fil dir
        cv2.imwrite( self.data_mat_root  + str(Image_ID) +".jpg", show1)
        # show the signal comparison in visdom
        if visdom_show_flag == True:
            x= np.arange(0, len(path))
            self.vis_ploter.plot_multi_arrays_append(x,path,title_name=str(Image_ID),legend = 'truth' )
            self.vis_ploter.plot_multi_arrays_append(x,path_deep,title_name=str(Image_ID),legend = 'Deep Learning' )
            self.vis_ploter.plot_multi_arrays_append(x,path_tradition,title_name=str(Image_ID),legend = 'Traditional' )
        # save comparison signals to matlab
        if Save_matlab_flag == True :
            self.matlab.buffer_4(Image_ID,path,path_deep,path_tradition)
            self.matlab.save_mat()
            pass

         


     def generate_NURD(self):
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
            if NURD_remove_shift_flag ==True:
                path= path- (np.mean(path) - Window_LEN/2 )

        
            # create the shifted image
            Shifted_IMG   = VIDEO_PEOCESS.de_distortion(original_IMG,path,Image_ID,0)
            if add_noise_flag == True:
                noise_type  =  str(self.noise_selector[int(Image_ID)%4])
                #noise_type = "gauss_noise"
                original_IMG  =  self.noisy(noise_type,original_IMG)
                Shifted_IMG  =  self.noisy(noise_type,Shifted_IMG)
            # save all the result

            cv2.imwrite(self.data_pair1_root  + str(Image_ID) +".jpg", original_IMG)
            cv2.imwrite(self.data_pair2_root  + str(Image_ID) +".jpg", Shifted_IMG)
            ## validation 
            #self.validation(original_IMG,Shifted_IMG,path,Image_ID) 

            #steam[Len_steam-1,:,:]  = original_IMG  # un-correct 
            #steam[Len_steam-2,:,:]  = Shifted_IMG  # correct 
            #Costmatrix,shift_used = COSTMtrix.matrix_cal_corre_full_version3_2GPU(original_IMG,Shifted_IMG,0) 
            #Costmatrix  = myfilter.gauss_filter_s (Costmatrix) # smooth matrix
            #show1 =  Costmatrix 
            #for i in range ( len(path)):
            #    show1[int(path[i]),i]=254
            #cv2.imwrite(self.data_mat_root  + str(Image_ID) +".jpg", show1)



            print ("[%s]   is processed. test point time is [%f] " % (read_id ,0.1))

            read_id +=1
     def generate_overall_shifting(self):
         #read one from the original
            #random select one IMG frome the oringinal 
        read_id = 0
        Len_steam =5
        #steam=np.zeros((Len_steam,self.H,self.W)) # create video buffer
        while (1):
            random_shifting = np.random.random_sample()*Overall_shiftting_WinLen
            #random_shifting = random.random() * Overall_shiftting_WinLen
            OriginalpathDirlist = os.listdir(self.original_root)    # 
            sample = random.sample(OriginalpathDirlist, 1)  # 
            Sample_path = self.original_root +   sample[0]
            original_IMG = cv2.imread(Sample_path)
            original_IMG  =   cv2.cvtColor(original_IMG, cv2.COLOR_BGR2GRAY)
            original_IMG = cv2.resize(original_IMG, (self.W,self.H), interpolation=cv2.INTER_AREA)
            #original_IMG = cv2.resize(original_IMG, (self.W,self.H), interpolation=cv2.INTER_AREA)
            H,W = original_IMG.shape

            #read the path and Image number from the signal file
            #get the Id of image which should be poibnt to
            Image_ID = int( self.path_DS.signals[Save_signal_enum.image_iD.value, read_id])
            #get the path
            path  = self.path_DS.path_saving[read_id,:]
            #change the signal too
            self.path_DS.path_saving[read_id,:] = path* 0 + random_shifting
            path =  signal.resample(path, W)*0 + random_shifting  #resample the path
            
            #resave the signal

            # create the shifted image
            Shifted_IMG   = VIDEO_PEOCESS.de_distortion(original_IMG,path,Image_ID,0)
            # save all the result
            cv2.imwrite(self.data_pair1_root  + str(Image_ID) +".jpg", original_IMG)
            cv2.imwrite(self.data_pair2_root  + str(Image_ID) +".jpg", Shifted_IMG)
            self.path_DS.save()
            #self.validation_shift(original_IMG,Shifted_IMG,path,Image_ID) 

            ## validation 
            #steam[Len_steam-1,:,:]  = original_IMG  # un-correct 
            #steam[Len_steam-2,:,:]  = Shifted_IMG  # correct 
            #Costmatrix,shift_used = COSTMtrix.matrix_cal_corre_full_version3_2GPU(original_IMG,Shifted_IMG,0) 
            ##Costmatrix  = myfilter.gauss_filter_s (Costmatrix) # smooth matrix
            #show1 =  Costmatrix 
            #for i in range ( len(path)):
            #    show1[int(path[i]),i]=254
            #cv2.imwrite(self.data_mat_root  + str(Image_ID) +".jpg", show1)



            print ("[%s]   is processed. test point time is [%f] " % (read_id ,0.1))

            read_id +=1

     # generate the   OCT iamge with combination of NURD and group shifting
     def generate_NURD_overall_shifting(self):
          #read one from the original
            #random select one IMG frome the oringinal 
        read_id = 0   # read pointer initialization
        Len_steam =5 # create the buffer for validation 
        steam=np.zeros((Len_steam,self.H,self.W)) # create video buffer
        while (1):
            #list all the picture for video generating, ensure the original folder has only one image
            OriginalpathDirlist = os.listdir(self.original_root) 
            sample = random.sample(OriginalpathDirlist, 1)  #  ramdom choose the name in folder list
            Sample_path = self.original_root +   sample[0] # create the reading path this radom picture
            original_IMG = cv2.imread(Sample_path) # get this image 
            original_IMG  =   cv2.cvtColor(original_IMG, cv2.COLOR_BGR2GRAY) # to gray
            original_IMG = cv2.resize(original_IMG, (self.W,self.H), interpolation=cv2.INTER_AREA)

            #read the path and Image number from the signal file
            #get the Id of image which should be poibnt to
            Image_ID = int( self.path_DS.signals[Save_signal_enum.image_iD.value, read_id])
            #get the path
            path  = self.path_DS.path_saving[read_id,:]
            if NURD_remove_shift_flag ==True:
                path= path- (np.mean(path) - Window_LEN/2 )
            #   exragene for diaplay
            path = (path -np.mean(path))*0.6+np.mean(path)
            path =  signal.resample(path, self.W)#resample the path
            overall_shifting = Image_ID 
            overall_shifting = min(overall_shifting,self.W/2) # limit the shifting here, maybe half the lenghth is sufficient  for the combination

            #Combine the overall shifting with NURD

            path = path  + overall_shifting
            # create the shifted image
            Shifted_IMG   = VIDEO_PEOCESS.de_distortion(original_IMG,path,Image_ID,0)

            # add noise to image pair for validation
            if add_noise_flag == True:
                noise_type  =  str(self.noise_selector[int(Image_ID)%4])
                #noise_type = "gauss_noise"
                original_IMG  =  self.noisy(noise_type,original_IMG)
                Shifted_IMG  =  self.noisy(noise_type,Shifted_IMG)

            # save all the result
            cv2.imwrite(self.data_pair1_root  + str(Image_ID) +".jpg", original_IMG)
            cv2.imwrite(self.data_pair2_root  + str(Image_ID) +".jpg", Shifted_IMG)

            # save generate information: the NURD and shift used for generating img pair to matlab
            if Save_matlab_flag == True :
                self.matlab.buffer_overall_shift_NURD(Image_ID,overall_shifting,path)
                self.matlab.save_mat_infor_of_over_allshift_with_NURD()
                self.matlab.save_pkl_infor_of_over_allshift_with_NURD()
                pass
            ## validation 
            self.validation_shift(original_IMG,Shifted_IMG,path,Image_ID) 

            #self.validation(original_IMG,Shifted_IMG,path,Image_ID) 

            #steam[Len_steam-1,:,:]  = original_IMG  # un-correct 
            #steam[Len_steam-2,:,:]  = Shifted_IMG  # correct 
            #Costmatrix,shift_used = COSTMtrix.matrix_cal_corre_full_version3_2GPU(original_IMG,Shifted_IMG,0) 
            #Costmatrix  = myfilter.gauss_filter_s (Costmatrix) # smooth matrix
            #show1 =  Costmatrix 
            #for i in range ( len(path)):
            #    show1[int(path[i]),i]=254
            #cv2.imwrite(self.data_mat_root  + str(Image_ID) +".jpg", show1)



            print ("[%s]   is processed. test point time is [%f] " % (read_id ,0.1))

            read_id +=1
 



if __name__ == '__main__':
        generator   = DATA_Generator()
        #generator.close_the_origin_signal()
        #save_test  = Save_Signal_matlab()
        #id,nurd,shift  =  save_test.read_pkl_infor_of_over_allshift_with_NURD()
 
        
        # generator.generate_NURD ()
        generator.generate_overall_shifting()
        #generator.generate_NURD_overall_shifting()