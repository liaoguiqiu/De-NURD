operatedir_matrix  =  "..\\saved_matrix\\"
 
operatedir_one =  "..\\saved_matrix\\126.jpg"
#operatedir_video = "E:\\PhD\\trying\\saved_original\\"
operatedir_video = "..\\saved_filtered_img\\"
savedir_path = "..\\saved_processed\\"

import cv2
import math
import numpy as np
from median_filter_special import  myfilter
import pandas as pd
import os
import torch
import scipy.signal as signal
from scipy.stats.stats import pearsonr   
import random
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
from  path_finding import PATH
from scipy.ndimage import gaussian_filter1d
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 

global shift_integral
class VIDEO_PEOCESS:
    def img_interpilate(image):
        h, w= image.shape
        for i in  range(h):
           s = pd.Series(image[i,:])
           image[i,:] = s.interpolate()
        return image

    def de_distortion(image,path,sequence_num):
        new= image
        h, w= image.shape
        new=new*0+np.nan
        long_3_img  = np.append(image,image,axis=1) # cascade
        long_3_img = np.append(long_3_img,image,axis=1) # cascade
        shift_diff= path - int(Window_LEN/2)
        global shift_integral
        if sequence_num==0:
           shift_integral = shift_diff*0
           shift_integral = shift_integral + shift_diff 
        else:
           shift_integral = shift_integral + shift_diff 
         
        
        for i in range ( len(shift_diff)):
            #limit the integral
            shift_integral[i]=max(min(shift_integral[i], 100), -100)
            #if (np.absolute(shift_integral[i])<5):
            #    shift_integral[i]=0
            new_position  = int(shift_integral[i]+i)
            if(new_position<0):
                new_position = w+new_position
            if (new_position>=w):
                new_position= new_position -w
            #new_position= max(min(new_position,  w-1), -w)
            #new[:,int(new_position)] = long_3_img[:,i +w]
            #new_position= max(min(new_position,  w-1), 0)
            new[:,int(new_position)] = image[:,i]
            
        interp_img = VIDEO_PEOCESS.img_interpilate(new)    
        return interp_img


    def correct_video( image,mat,sequence_num):
       

        

         
     
        start_point= PATH.find_the_starting(mat)
        path1,path_cost1=PATH.search_a_path(mat,start_point)
        path1 = gaussian_filter1d(path1,3)  

        img_corrected = VIDEO_PEOCESS.de_distortion(image,path1,sequence_num)

        show1 =  mat
        #circular = cv2.linearPolar(img_corrected, (img_corrected.shape[0]/2, img_corrected.shape[1]/2), 
        #                           462, cv2.WARP_INVERSE_MAP)
        new_frame=cv2.rotate(img_corrected,rotateCode = 2) 
        circular = cv2.linearPolar(new_frame, (new_frame.shape[1]/2 , new_frame.shape[0]/2), 
                                   200, cv2.WARP_INVERSE_MAP)
        cv2.imwrite(savedir_path  + str(sequence_num) +".jpg", circular)
  

        for i in range ( len(path1)):
            show1[int(path1[i]),i]=254
 
        cv2.imshow('step_process',show1) 
        cv2.imshow('original video',image) 
        cv2.imshow('correcr video',img_corrected.astype(np.uint8))
        cv2.imshow('rota video',new_frame.astype(np.uint8))
        cv2.imshow('circular video',circular.astype(np.uint8)) 
        return 0


#jiust check result of one pic 
#frame = cv2.imread(operatedir_one)
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#filtered_gray  = myfilter.gauss_filter_s (gray)
#VIDEO_PEOCESS.correct_video(filtered_gray,filtered_gray,1)

read_sequence = os.listdir(operatedir_matrix)

seqence_Len = len(read_sequence)
for i in range(seqence_Len):
#for i in os.listdir("E:\\estimagine\\vs_project\\PythonApplication_data_au\\pic\\"):
 
     
   
     
        #read matrix
        Matri_path = operatedir_matrix + str(i+10) + ".jpg"
        img = cv2.imread(Matri_path)
        gray  =   cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        Matrix  = myfilter.gauss_filter_s (gray)
        # read imag for process 
        img_path = operatedir_video + str(i+10)+ ".jpg"
        video = cv2.imread(img_path)
        gray_video  =   cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
        


        VIDEO_PEOCESS.correct_video(gray_video,Matrix,int(i))
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
