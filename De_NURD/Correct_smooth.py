operatedir_matrix  =  "../../saved_matrix/"
operatedir_matrix_unprocessed  =  "../../saved_matrix_unprocessed/"

 #saved_matrix_unprocessed
operatedir_one =  "../../saved_matrix/126.jpg"
operatedir_video = "../../saved_original/"
#operatedir_video = "E:/PhD/trying/saved_filtered_img/"
savedir_path = "../../saved_processed/"
savedir_rectan_ = "../../saved_processed_polar/"

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
#from matplotlib.pyplot import *
#from mpl_toolkits.mplot3d import Axes3D
from  path_finding import PATH
from median_filter_special import  myfilter
from cost_matrix import  COSTMtrix
from cost_matrix import Window_LEN ,Overall_shiftting_WinLen
from scipy.ndimage import gaussian_filter1d
from time import time
import scipy.io
from parrallel_thread import Dual_thread_Overall_shift_NURD
from shift_deploy import Shift_Predict
from  basic_trans import Basic_oper
from Path_post_pcs import PATH_POST
from ekf import EKF
myekf = EKF()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Resample_size =Window_LEN
Path_length = 128
#read_start = 100
read_start = 0

Debug_flag  = True
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
    def de_distortion_overall_shifting(image,path,sequence_num,addition_window_shift):
        new= image
        h, w= image.shape
        new=new*0  # Nan pixel will be filled by intepolation processing
        mask =  new  + 255
        shift_diff= path +addition_window_shift - int(Overall_shiftting_WinLen/2)  # additional compensation 
        shift_integral = shift_diff # not += : this is iteration way
        #every line will be moved to a new postion
        for i in range ( len(shift_diff)):
            #limit the integral             
            new_position  = int(shift_integral[i]+i)
            # deal with the boundary exceeding
            if(new_position<0):
                new_position = w+new_position
            if (new_position>=w):
                new_position= new_position -w
            #move this line to new position
            new[:,int(new_position)] = image[:,i]
            mask[:,int(new_position)] = 0

        # connect the statrt and end before the interpilate          
        #modified to  # connect the statrt and end before the interpilate
        long_3_img  = np.append(new,new,axis=1) 
        long_3_img = np.append(long_3_img,new,axis=1) # cascade
        longmask  = np.append(mask,mask,axis=1) 
        longmask  = np.append(longmask,mask,axis=1) 

        interp_img=cv2.inpaint(long_3_img, longmask, 2, cv2.INPAINT_TELEA)
        #interp_img = VIDEO_PEOCESS.img_interpilate(long_3_img) # interpolate by row
        new= interp_img[:,w:2*w] # take the middle one 
        return new
#----------------------#

#----------------------#
# apply the path (shift) and shift compensation to correct
    def de_distortion_integral(image,shift_integral,sequence_num):
        new= image
        h, w= image.shape
        new=new*0  # Nan pixel will be filled by intepolation processing
        mask =  new  + 255
       

        #shift_integral = shift_integral + shift_diff.astype(int) # not += : this is iteration way
        #shift_integral = np.clip(shift_integral, - 35,35)
        #every line will be moved to a new postion
        for i in range ( len(shift_integral)):
            #limit the integral             
            new_position  = int(shift_integral[i]+i)
            # deal with the boundary exceeding
            if(new_position<0):
                new_position = w+new_position
            if (new_position>=w):
                new_position= new_position -w
            #move this line to new position
            new[:,int(new_position)] = image[:,i]
            mask[:,int(new_position)] = 0

        # connect the statrt and end before the interpilate          
        #modified to  # connect the statrt and end before the interpilate
        long_3_img  = np.append(new,new,axis=1) 
        long_3_img = np.append(long_3_img,new,axis=1) # cascade
        longmask  = np.append(mask,mask,axis=1) 
        longmask  = np.append(longmask,mask,axis=1) 

        # interp_img=cv2.inpaint(long_3_img, longmask, 2, cv2.INPAINT_TELEA)
        interp_img=cv2.inpaint(new, mask, 1, cv2.INPAINT_TELEA)
        # the time cmcumption of this is 0.02s
        

        #interp_img = VIDEO_PEOCESS.img_interpilate(long_3_img) # interpolate by row
        # new= interp_img[:,w:2*w] # take the middle one 
        new= interp_img  

        return new
#----------------------#
#----------------------#
# 2 use the interpolate inv path
    def de_distortion_integral2(image,shift_integral,sequence_num):
        #new= image
        h, w= image.shape
        #new=new*0  # Nan pixel will be filled by intepolation processing
        #mask =  new  + 255
       
        add_3_img  = np.append(image,image,axis=1) # cascade
        add_3_img = np.append(add_3_img,image,axis=1) # cascade
        new= add_3_img*0

        #shift_integral = shift_integral + shift_diff.astype(int) # not += : this is iteration way
        #shift_integral = np.clip(shift_integral, - 35,35)
        #every line will be moved to a new postion
        add_3   = np.append(shift_integral ,shift_integral,axis=0) # cascade
        add_3   = np.append(add_3,shift_integral,axis=0) # cascade
        path_inv  = add_3*np.nan

        for i in range ( len(add_3)):
            #limit the integral             
            new_position  = add_3[i]+i
            # deal with the boundary exceeding
            if(new_position<0):
                new_position = 0
            if (new_position>=3*w):
                new_position= 3*w-1
            #move this line to new position
            path_inv[int(new_position)] = float(i)
            #mask[:,int(new_position)] = 0
        #add_3   = np.append(path_inv[::-1],path_inv,axis=0) # cascade
        #add_3   = np.append(add_3,path_inv[::-1],axis=0) # cascade
        s = pd.Series(path_inv)
        path_inv = s.interpolate()
        #path_inv = path_inv[w:2*w].to_numpy() 
      
        for i in range ( w,2*w):
            #limit the integral             
            new_position  = int(path_inv[i])
            # deal with the boundary exceeding
            if(new_position<=0):
                new_position = 0
            if (new_position>=3*w):
                new_position= 3*w-1
            #move this line to new position
            new[:,i] = add_3_img[:,new_position]
            #mask[:,int(new_position)] = 0
        # connect the statrt and end before the interpilate          
        #modified to  # connect the statrt and end before the interpilate
        #long_3_img  = np.append(new,new,axis=1) 
        #long_3_img = np.append(long_3_img,new,axis=1) # cascade
        #longmask  = np.append(mask,mask,axis=1) 
        #longmask  = np.append(longmask,mask,axis=1) 

        ## interp_img=cv2.inpaint(long_3_img, longmask, 2, cv2.INPAINT_TELEA)
        #interp_img=cv2.inpaint(new, mask, 1, cv2.INPAINT_TELEA)
        ## the time cmcumption of this is 0.02s
        

        ##interp_img = VIDEO_PEOCESS.img_interpilate(long_3_img) # interpolate by row
        short= new[:,w:2*w] # take the middle one 
        #new= interp_img  

        return short
#----------------------#

#----------------------#
#correct and save /display results
    def correct_video_with_shifting( image,corre_shifting,sequence_num,addition_window_shift):
        H,W= image.shape  #get size of image
 
        path1  = np.zeros(W)
   
        path1 = corre_shifting + path1      
        path_cost1  = 0       
        img_corrected = VIDEO_PEOCESS.de_distortion_overall_shifting(image,path1,sequence_num,addition_window_shift)
        return img_corrected,path1,path_cost1

#----------------------#
    def get_warping_vextor(mat):
        H,W= mat.shape  #get size of image
        show1 =  mat.astype(float)
        path1  = np.zeros(W)
        #small the initial to speed path finding 
        #mat = cv2.resize(mat, (Resample_size,H), interpolation=cv2.INTER_AREA)
        #mat = cv2.resize(mat, (Resample_size,Resample_size), interpolation=cv2.INTER_AREA)


        #start_point= PATH.find_the_starting(mat) # starting point for path searching
        ##middle_point  =  PATH.calculate_ave_mid(mat)
        #path1,path_cost1=PATH.search_a_path(mat,start_point) # get the path and average cost of the path
        path1,path_cost1=PATH.search_a_path_deep_multiscal_small_window_fusion2(mat) # get the path and average cost of the path
       
        #path1 = corre_shifting + path1
       
        #path1 =0.5 * path1 + 0.5 * corre_shifting 
        path_cost1  = 0
        path1 = gaussian_filter1d(path1,3) # smooth the path 
        return path1
#----------------------#
#----------------------#
    def fusion_estimation( shift_integral,path,overall_shift,I):
        #overall_shift =0
        shift_diff= path - int(Window_LEN/2)  # additional compensation 
        ##shift_diff = gaussian_filter1d(shift_diff,3) # smooth the path 

        # PI fusion
        shift_integral = shift_integral + shift_diff  # not += : this is iteration way
        #shift_integral = PATH_POST.path_integral(shift_integral,shift_diff)

        shift_integral = shift_integral - 0.15*(shift_integral-overall_shift) - 0.000001* I
        # EKF fusion
        #shift_integral = myekf.update(shift_diff,overall_shift)


        #shift_integral = np.clip(shift_integral,overall_shift- Window_LEN/2,overall_shift+ Window_LEN/2)
        #shift_integral = gaussian_filter1d(shift_integral,5) # smooth the path 

        #shift_integral = shift_integral - 0.2*(shift_integral-overall_shift) - 0.0000001*I
        #shift_integral = shift_integral*0 + overall_shift  

        #shift_integral = gaussian_filter1d(shift_integral,3) # smooth the path 

        shift_integral = gaussian_filter1d(shift_integral,10) # smooth the path 
       
        return shift_integral
#----------------------#
#correct and save /display results
    def correct_video( image,corre_shifting,mat,shift_integral,sequence_num,addition_window_shift):
        H,W= mat.shape  #get size of image
        show1 =  mat.astype(float)
        path1  = np.zeros(W)
        #small the initial to speed path finding 
        #mat = cv2.resize(mat, (Resample_size,H), interpolation=cv2.INTER_AREA)
        #mat = cv2.resize(mat, (Resample_size,Resample_size), interpolation=cv2.INTER_AREA)


        start_point= PATH.find_the_starting(mat) # starting point for path searching
        ##middle_point  =  PATH.calculate_ave_mid(mat)
        path1,path_cost1=PATH.search_a_path(mat,start_point) # get the path and average cost of the path
        #path1,path_cost1=PATH.search_a_path_deep_multiscal_small_window_fusion2(mat) # get the path and average cost of the path
       
        #path1 = corre_shifting + path1
       
        #path1 =0.5 * path1 + 0.5 * corre_shifting 
        path_cost1  = 0
        path1 = gaussian_filter1d(path1,3) # smooth the path 
        #path1 = path1 -(np.mean(path1) - int(Window_LEN/2)) # remove the meaning shifting

        #long_out  = np.append(np.flip(path1),path1) # flip to make the the start point and end point to be perfect interpolit
        #long_out  = np.append(long_out, np. flip ( path1))
        #long_out = gaussian_filter1d (long_out ,1)
        #long_path_upsam  = signal.resample(long_out, 3*W)
        #path_upsam = long_path_upsam[W:2*W]
        #path1  = path_upsam /Resample_size * H

        #path1,path_cost1=PATH.search_a_path_GPU (mat) # get the path and average cost of the path
       
       
        
        
        # applying the correct
        img_corrected,shift_integral = VIDEO_PEOCESS.de_distortion_integral (image,path1,shift_integral,sequence_num,addition_window_shift)


        
        


        #cv2.imshow('step_process',show1.astype(np.uint8)) 
        #cv2.imshow('original video',image) 
        #cv2.imshow('correcr video',img_corrected.astype(np.uint8))
        #cv2.imshow('rota video',new_frame.astype(np.uint8))
        #cv2.imshow('circular video',circular.astype(np.uint8)) 
        return img_corrected,path1,shift_integral,path_cost1

#----------------------#
#----------------------#

#jiust check result of one pic 
#frame = cv2.imread(operatedir_one)
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#filtered_gray  = myfilter.gauss_filter_s (gray)
#VIDEO_PEOCESS.correct_video(filtered_gray,filtered_gray,1)


#---------main schedule-------------#
    def main():

        read_sequence = os.listdir(operatedir_video) # read all file name
        seqence_Len = len(read_sequence)    # get all file number 
        img_path = operatedir_video +  str(read_start) +".jpg"
        video = cv2.imread(img_path)  #read the first one to get the image size
        gray_video  =   cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
        H_ori , W_ori  = gray_video.shape
        gray_video = cv2.resize(gray_video, (832,H_ori), interpolation=cv2.INTER_LINEAR)
        shift_integral = np.zeros(832)
        

        Len_steam =2
        H,W= gray_video.shape  #get size of image
        H_start = 0
        H_end = int(H)
        steam=np.zeros((Len_steam,H_end-H_start,W))
        steam2=np.zeros((Len_steam,H_end-H_start ,W))
        save_sequence_num = 0  # processing iteration initial 
        addition_window_shift=0 # innitial shifting parameter
        Window_ki_error = 0
        Window_kp_error = 0
        Kp=0 # initial shifting paramerter
        dual_thread  = Dual_thread_Overall_shift_NURD()
        for sequence_num in range(read_start,seqence_Len):
        #for i in os.listdir("E:/estimagine/vs_project/PythonApplication_data_au/pic/"):
                # read imag for process 
                img_path = operatedir_video + str(sequence_num+0)+ ".jpg" # starting from 10
                video = cv2.imread(img_path)
                gray_video  =   cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
                H_ori , W_ori  = gray_video.shape
                gray_video = cv2.resize(gray_video, (832,H_ori), interpolation=cv2.INTER_LINEAR)
                start_time  = time()

                if(sequence_num<read_start+ 5):
                    # bffer a resized one to coputer the path and cost matrix
                    steam=np.append(steam,[gray_video[H_start:H_end,:] ],axis=0) # save sequence3
                    # normal beffer process
                    steam= np.delete(steam , 0,axis=0)
      

                    steam2=np.append(steam2,[gray_video[H_start:H_end,:] ],axis=0) # save sequence
                    steam2= np.delete(steam2 , 0,axis=0)
                else:
                    steam=np.append(steam,[gray_video[H_start:H_end,:] ],axis=0) # save sequence
                    # no longer delete the fist  one
                    steam= np.delete(steam , 1,axis=0)
                    steam2=np.append(steam2,[gray_video[H_start:H_end,:] ],axis=0) # save sequence
                    steam2= np.delete(steam2 , 0,axis=0)

                    Corrected_img  = myfilter.steam_filter_avg (steam2)
                     
                     

                    path_cost =0
                    #overall_shifting3,shift_used3 = COSTMtrix.Img_fully_shifting_correlation(Corrected_img[H_start:H_end,:],
                    #                                          steam[0,:,:],  0) 
                    #Corrected_img,path,path_cost=   VIDEO_PEOCESS.correct_video_with_shifting(gray_video,overall_shifting,int(sequence_num),shift_used1 )

                    # remove the central shifting 
                    #addition_window_shift = -0.00055*(np.mean(path)- int(Window_LEN/2))+addition_window_shift
                   
                
            
                    # remove intergral bias ( here just condsider the overal img should be in the center) 
          

                    #correct method 1
                    #shift_integral = shift_integral - 1*(np.mean(shift_integral)-addition_window_shift) -  Window_ki_error

                    #shift_integral = shift_integral - 0.1 * np.mean(shift_integral)

                    #corre method 2

                    #re！！！！！Next time remenber to remove the un-corrected image from the stream

                    #save the  corrected result for group shifting  
                    steam=np.append(steam,[Corrected_img[H_start:H_end,:] ],axis=0) # save sequence
                    # no longer delete the fist  one
                    steam= np.delete(steam , 1,axis=0)

                    #steam2=np.append(steam2,[Corrected_img ],axis=0) # save sequence
                    ## no longer delete the fist  one
                    #steam2= np.delete(steam2 , 0,axis=0)
                     
                     
                    circular = Basic_oper.tranfer_frome_rec2cir(Corrected_img)
       

                    cv2.imwrite(savedir_path  + str(sequence_num) +".jpg", circular)
            
                    cv2.imwrite(savedir_rectan_  + str(sequence_num) +".jpg",Corrected_img )

                    print ("[%s]   is processed.   " % (sequence_num ))
                    test_time_point = time()
                    print (" all test point time is [%f] " % ( test_time_point - start_time))

if __name__ == '__main__':
    VIDEO_PEOCESS.main()