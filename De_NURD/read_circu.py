#operatedir_pic =  "../initialbackground/"
#operatedir_video =  "D:/PhD/trying/OCT/P-ID_Name_25092019160318VIDEO.avi"
#operatedir_video =  "../../OCT/P-ID_Name_25092019160318VIDEO.avi"
#operatedir_video =  "../../OCT/P-ID_Name_25092019161813-7500rpm-G1_0.05_4_25_extracted.avi"
#E:\PhD\trying\OCT\OCT aligment
#operatedir_video =  "../../OCT/OCT aligment/22JAN2020AUTO_01.avi"
#operatedir_video =  "../../OCT/OCT aligment/phantom-01_2412020121234.avi"
#operatedir_video =  "../../OCT/new video/P-ID_Name_25092019164030.avi"
#D:\PhD\trying\tradition_method\OCT\database_download\cardiovascular
operatedir_video =  "../../OCT/database_download/cardiovascular/1.mp4"
operatedir_video =  "../../OCT/database_download/cardiovascular/2.mp4"

#operatedir_video =  "../../OCT/database_download/Lungs/1.mov"

 

#phantom-01_2412020121234
#operatedir_video =  "../../OCT/P-ID_Name_25092019160318VIDEO.avi"
savedir_matrix  = "../../saved_matrix/"
savedir_original  = "../../saved_original/"
savedir_filtered_OCT  = "../../saved_filtered_img/"
savedir_original_circular = "../../saved_original_circular/"
crop_flag  =True
reverse_flag =False
#used python packages
import cv2
import math
import numpy as np
import os
import random
#from matplotlib.pyplot import *
#from mpl_toolkits.mplot3d import Axes3D
from median_filter_special import  myfilter
from cost_matrix import  COSTMtrix
Down_sample_CNT= 3
#PythonETpackage for xml file edition
try: 
    import xml.etree.cElementTree as ET 
except ImportError: 
    import xml.etree.ElementTree as ET 
import sys 


def tranfer_frome_cir2rec(gray):
    H,W = gray.shape
    value = np.sqrt(((H/2.0)**2.0)+((W/2.0)**2.0))

    polar_image = cv2.linearPolar(gray,(W/2, H/2), value, cv2.WARP_FILL_OUTLIERS)

    polar_image = polar_image.astype(np.uint8)
    polar_image=cv2.rotate(polar_image,rotateCode = 0) 
    return polar_image
def tranfer_frome_rec2cir(gray):
    H,W = gray.shape
    value = np.sqrt(((W/2.0)**2.0)+((W/2.0)**2.0))
    value/=2
    grayr=cv2.rotate(gray,rotateCode = 2) 
    H,W = grayr.shape

    #value = 200
    #circular = cv2.linearPolar(new_frame, (new_frame.shape[1]/2 , new_frame.shape[0]/2), 
    #                               200, cv2.WARP_INVERSE_MAP)
    #circular = cv2.linearPolar(gray,(W/2, H/2), value, cv2.WARP_INVERSE_MAP)
    circular = cv2.linearPolar(grayr,(W/2, H/2), value, cv2.WARP_INVERSE_MAP)


    circular = circular.astype(np.uint8)
    #polar_image=cv2.rotate(polar_image,rotateCode = 0) 
    return circular

if __name__ == '__main__':
    #GPU acceleration
    #from numba import vectorize
    #from numba import jit
    #@jit
    #def multiply(a, b):
    #    return a * b /255.0
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = [] 

    #algorithm start
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(operatedir_video)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    #fig = figure() 
    # Read until video is completed
    Len_steam =5
    ret, frame = cap.read()
    if ret == True:
        H,W,_ = frame.shape
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    H_start = 0
    H_end = H
    W_start = int(W/2)
    #W_end = int(W/2)
    W_end = W
     
    steam=np.zeros((Len_steam,H_end-H_start,W))
    steam2=np.zeros((Len_steam,H_end-H_start,W))
    save_sequence_num = 0

    read_start  =  0
    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
      read_start += 1
      if read_start % Down_sample_CNT !=0:
          continue
      if ret == True:
        
        # Display the resulting frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H,W = gray.shape
        H_start = 0
        H_end = H
        W_start = int(W/2)
        #W_end = int(W/2)
        W_end = W
        gray  = gray [H_start:H_end, W_start: W_end]
        #H,W = gray.shape
         
        if reverse_flag == True:
            gray = 255  - gray
   
        #crop_H_test = gray[H_start:H_end,:] 
        
        #filter_img = crop_H_test
        
        #filter_img= myfilter.gauss_filter_s(crop_H_test)
    
        #steam=np.append(steam,[filter_img],axis=0) # save sequence
        #steam= np.delete(steam , 0,axis=0) 
    
        #Costmatrix = COSTMtrix.matrix_cal_corre_full_version(steam)
    

        #cv2.imshow('initial',frame)
        #cv2.imshow('gray',gray)
        #cv2.imshow('slice_vertical',crop_V_test)
        #cv2.imshow('slice_horizental',crop_H_test)
        #cv2.imshow('steamfilter',stea_filter.astype(np.uint8))
        #cv2.imshow('ffilter',filter_img)
        #cv2.imshow('costmatrix',Costmatrix.astype(np.uint8))
        value = np.sqrt(((H/2.0)**2.0)+((W/2.0)**2.0))

        polar_image = tranfer_frome_cir2rec(gray)

        H,W= polar_image.shape
        H_start = 0
        H_end = H
        W_start = 0
        #W_end = int(W/2)
        W_end = W
        if crop_flag == True :
            polar_image = polar_image[H_start:H_end, W_start: W_end] 

        circular  = tranfer_frome_rec2cir(polar_image)
        
        #H,W= new_frame2.shape
        #circular3=np.ones((H,W))
        #circular2=circular3.astype(float)
        #circular = circular2*2
        #circular = cv2.linearPolar(new_frame3, (int(W/2) , int(H/2)),400, cv2.WARP_INVERSE_MAP)
        #circular=circular.astype(np.uint8)

        #cv2.imwrite(savedir_matrix  + str(save_sequence_num) +".jpg", Costmatrix)
        cv2.imwrite(savedir_original  + str(save_sequence_num) +".jpg", polar_image)
        #cv2.imwrite(savedir_filtered_OCT  + str(save_sequence_num) +".jpg", filter_img)
        cv2.imwrite(savedir_original_circular  + str(save_sequence_num) +".jpg", circular)

        save_sequence_num+=1
        print ("[%s]   is processed." % (save_sequence_num))
    
 
      # Break the loop
      else: 
        break
 
    # When everything done, release the video capture object
    cap.release()
 
    # Closes all the frames
    cv2.destroyAllWindows()
