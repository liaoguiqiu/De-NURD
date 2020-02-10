operatedir_matrix  =  "..\\..\\saved_matrix\\"
operatedir_matrix_unprocessed  =  "..\\..\\saved_matrix_unprocessed\\"

 #saved_matrix_unprocessed
operatedir_one =  "..\\..\\saved_matrix\\126.jpg"
operatedir_video = "..\\..\\saved_original\\"
#operatedir_video = "E:\\PhD\\trying\\saved_filtered_img\\"
savedir_path = "..\\..\\saved_processed\\"
operatedir_cor_pair1 = "..\\..\\saved_pair1\\"
operatedir_cor_pair2 = "..\\..\\saved_pair2\\"
operatedir_cor_pair3 = "..\\..\\saved_pair3\\"

from analy import Save_signal_flag
# notificatiton for the naming of stream： all “steam” in this project means 
#"stream"

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
from median_filter_special import  myfilter
from cost_matrix import  COSTMtrix
from cost_matrix import Window_LEN 
from scipy.ndimage import gaussian_filter1d
from time import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Resample_size =Window_LEN
Path_length = 128
global shift_integral # integral (not interation) correction shift
global intergral_flag
intergral_flag =0

if (Save_signal_flag == True):
    from analy import MY_ANALYSIS
    from analy import Save_signal_enum
    signal_saved = MY_ANALYSIS()

class VIDEO_PEOCESS:

#----------------------#
#intepolate one image by rows
    def img_interpilate(image):
        h, w= image.shape
        for i in  range(h):
           s = pd.Series(image[i,:])
           image[i,:] = s.interpolate()
        return image
#----------------------#

#----------------------#
# apply the path (shift) and shift compensation to correct
    def de_distortion(image,path,sequence_num,addition_window_shift):
        new= image
        h, w= image.shape
        new=new*0+np.nan # Nan pixel will be filled by intepolation processing
        shift_diff= path +addition_window_shift - int(Window_LEN/2)  # additional compensation 
        shift_integral = shift_diff # not += : this is iteration way
        shift_integral  = np.append(shift_integral,shift_diff,axis=0) 
        shift_integral  = np.append(shift_integral,shift_diff,axis=0) 
        long_3_img  = np.append(image,image,axis=1) 
        long_3_img = np.append(long_3_img,image,axis=1) # cascade
        longH, longW= long_3_img.shape
        corrected=long_3_img*0 #+np.nan # Nan pixel will be filled by intepolation processing
        mask   = long_3_img*0+255
        #every line will be moved to a new postion
        for i in range ( len(shift_integral)):
            #limit the integral             
            new_position  = int(shift_integral[i]+i)
            # deal with the boundary exceeding
            if(new_position<0):
                new_position = longW+new_position
            if (new_position>=longW):
                new_position= new_position -longW
            #move this line to new position
            corrected[:,int(new_position)] = long_3_img[:,i]
            mask[:,int(new_position)] = long_3_img[:,i]*0
        # connect the statrt and end before the interpilate          
        #modified to  # connect the statrt and end before the interpilate
        interp_img = cv2.inpaint(corrected.astype(np.uint8),
                                 mask.astype(np.uint8),2,cv2.INPAINT_TELEA) # interpolate by row
        #interp_img = corrected
        #interp_img = VIDEO_PEOCESS.img_interpilate(corrected) # interpolate by row
        new= interp_img[:,w:2*w] # take the middle one 
        return new
#----------------------#

#----------------------#
#correct and save /display results
    def correct_video( image,mat,sequence_num,addition_window_shift):
        H,W= mat.shape  #get size of image
        show1 =  mat.astype(float)
        #small the initial to speed path finding 
        #mat = cv2.resize(mat, (Resample_size,H), interpolation=cv2.INTER_AREA)
        #mat = cv2.resize(mat, (Resample_size,Resample_size), interpolation=cv2.INTER_AREA)

        #long_out  = np.append(np.flip(path1),path1) # flip to make the the start point and end point to be perfect interpolit
        #long_out  = np.append(long_out, np. flip ( path1))
        #long_out = gaussian_filter1d (long_out ,1)
        #long_path_upsam  = signal.resample(long_out, 3*W)
        #path_upsam = long_path_upsam[W:2*W]
        #path1  = path_upsam /Resample_size * H


        start_point= PATH.find_the_starting(mat) # starting point for path searching
        path1,path_cost1=PATH.search_a_path(mat,start_point) # get the path and average cost of the path
        path1 = gaussian_filter1d(path1,3) # smooth the path 

        #path1,path_cost1=PATH.search_a_path_GPU (mat) # get the path and average cost of the path
       
       
        
        
        # applying the correct
        img_corrected = VIDEO_PEOCESS.de_distortion(image,path1,sequence_num,addition_window_shift)

        
        new_frame=cv2.rotate(img_corrected,rotateCode = 2) 
        circular = cv2.linearPolar(new_frame, (new_frame.shape[1]/2 , new_frame.shape[0]/2), 
                                   200, cv2.WARP_INVERSE_MAP)
        for i in range ( len(path1)):
            show1[int(path1[i]),i]=254
        cv2.imwrite(savedir_path  + str(sequence_num) +".jpg", circular)
        cv2.imwrite(operatedir_matrix_unprocessed  + str(sequence_num) +".jpg", mat)
        cv2.imwrite(operatedir_matrix  + str(sequence_num) +".jpg", show1)

        #cv2.imshow('step_process',show1.astype(np.uint8)) 
        #cv2.imshow('original video',image) 
        #cv2.imshow('correcr video',img_corrected.astype(np.uint8))
        #cv2.imshow('rota video',new_frame.astype(np.uint8))
        #cv2.imshow('circular video',circular.astype(np.uint8)) 
        return img_corrected,path1,path_cost1

#----------------------#
#----------------------#

#jiust check result of one pic 
#frame = cv2.imread(operatedir_one)
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#filtered_gray  = myfilter.gauss_filter_s (gray)
#VIDEO_PEOCESS.correct_video(filtered_gray,filtered_gray,1)
    def main():
        #---------main schedule-------------#

        read_sequence = os.listdir(operatedir_video) # read all file name
        seqence_Len = len(read_sequence)    # get all file number 
        img_path = operatedir_video +   "0.jpg"
        video = cv2.imread(img_path)  #read the first one to get the image size
        gray_video  =   cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
        Len_steam =5
        H,W= gray_video.shape  #get size of image
        steam=np.zeros((Len_steam,H,W)) # create video buffer
        steam2=np.zeros((Len_steam,H,W)) # create video buffer
        save_sequence_num = 0  # processing iteration initial 
        addition_window_shift=0 # innitial shifting parameter
        Kp=0 # initial shifting paramerter
        for i in range(seqence_Len):
        #for i in os.listdir("E:\\estimagine\\vs_project\\PythonApplication_data_au\\pic\\"):
                start_time  = time()
                # read imag for process 
                img_path = operatedir_video + str(i+10)+ ".jpg" # starting from 10
                video = cv2.imread(img_path)
                gray_video  =   cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
                if(i<Len_steam):
                    steam=np.append(steam,[gray_video],axis=0) # save sequence
                    steam= np.delete(steam , 0,axis=0)
                    steam2=np.append(steam2,[gray_video],axis=0) # save sequence
                    steam2= np.delete(steam2 , 0,axis=0)
                else:
                    steam=np.append(steam,[gray_video],axis=0) # save sequence
                    steam= np.delete(steam , 0,axis=0)
                    steam2=np.append(steam2,[gray_video],axis=0) # save sequence
                    steam2= np.delete(steam2 , 0,axis=0)
                    # shifting used is zero in costmatrix caculation

                    ###########3
                    #save the pair o imag for deeping train
                    cv2.imwrite(operatedir_cor_pair1  + str(i) +".jpg", steam[Len_steam-1,:,:])
                    cv2.imwrite(operatedir_cor_pair2  + str(i) +".jpg", steam[Len_steam-2,:,:])
                    ########

                    #Costmatrix,shift_used = COSTMtrix.matrix_cal_corre_full_version_2(steam,0) 
                    Costmatrix,shift_used = COSTMtrix.matrix_cal_corre_full_version_2GPU(steam,0) 
                    #Costmatrix,shift_used = COSTMtrix.matrix_cal_Euler_GPU(steam,0) 

                    Costmatrix  = myfilter.gauss_filter_s (Costmatrix) # smooth matrix

                    #get path and correct image
                    Corrected_img,path,path_cost=   VIDEO_PEOCESS.correct_video(gray_video,Costmatrix,int(i),addition_window_shift +Kp )
                    # remove the central shifting 
                    addition_window_shift = -0.00055*(np.mean(path)- int(Window_LEN/2))+addition_window_shift
                    path_mean_error = (np.mean(path)- int(Window_LEN/2))
                    Kp = -0.055* path_mean_error
                    #re！！！！！Next time remenber to remove the un-corrected image from the stream
                    #steam=np.append(steam,[Corrected_img],axis=0) # save sequence
                    #steam= np.delete(steam , 0,axis=0)

                    if(Save_signal_flag==True):
      
                        new = np.zeros((signal_saved.DIM,1))
                        new[Save_signal_enum.image_iD.value] = i
                        new[Save_signal_enum.additional_kp.value]=  Kp
                        new[Save_signal_enum.additional_ki.value]=  addition_window_shift
                        new[Save_signal_enum.path_cost.value]=  path_cost
                        new[Save_signal_enum.mean_path_error.value]=  path_mean_error
                        signal_saved.add_new_iteration_result(new,path)
                        signal_saved.display_and_save2(i,new)
                    test_time_point = time()

                    print ("[%s]   is processed. test point time is [%f] " % (i ,test_time_point - start_time))
 
#VIDEO_PEOCESS.main()