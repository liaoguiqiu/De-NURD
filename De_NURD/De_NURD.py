#operatedir_pic =  "..\\initialbackground\\"
#operatedir_video =  "D:\\PhD\\trying\\OCT\\P-ID_Name_25092019160318VIDEO.avi"
#operatedir_video =  "..\\..\\OCT\\P-ID_Name_25092019160318VIDEO.avi"
#operatedir_video =  "..\\..\\OCT\\P-ID_Name_25092019164030.avi"
#E:\PhD\trying\OCT\OCT aligment
operatedir_video =  "..\\..\\OCT\\OCT aligment\\22JAN2020AUTO_02_-Scale-0_5-41pix_Per_mm.avi"

#operatedir_video =  "..\\..\\OCT\\P-ID_Name_25092019161813-7500rpm-G1_0.05_4_25_extracted.avi"
savedir_matrix  = "..\\..\\saved_matrix\\"
savedir_original  = "..\\..\\saved_original\\"
savedir_filtered_OCT  = "..\\..\\saved_filtered_img\\"

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

#PythonETpackage for xml file edition
try: 
    import xml.etree.cElementTree as ET 
except ImportError: 
    import xml.etree.ElementTree as ET 
import sys 
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
H_start = 0
H_end = W
 
steam=np.zeros((Len_steam,H_end-H_start,W))
steam2=np.zeros((Len_steam,H_end-H_start,W))
save_sequence_num = 0
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
    # Display the resulting frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    crop_H_test = gray[H_start:H_end,:] 
    filter_img = crop_H_test
    
    filter_img= myfilter.gauss_filter_s(crop_H_test)
    
    steam=np.append(steam,[filter_img],axis=0) # save sequence
    steam= np.delete(steam , 0,axis=0) 
    
    #Costmatrix = COSTMtrix.matrix_cal_corre_full_version(steam)
    

    #cv2.imshow('initial',frame)
    #cv2.imshow('gray',gray)
    #cv2.imshow('slice_vertical',crop_V_test)
    #cv2.imshow('slice_horizental',crop_H_test)
    #cv2.imshow('steamfilter',stea_filter.astype(np.uint8))
    #cv2.imshow('ffilter',filter_img)
    #cv2.imshow('costmatrix',Costmatrix.astype(np.uint8))


    #cv2.imwrite(savedir_matrix  + str(save_sequence_num) +".jpg", Costmatrix)
    cv2.imwrite(savedir_original  + str(save_sequence_num) +".jpg", crop_H_test)
    cv2.imwrite(savedir_filtered_OCT  + str(save_sequence_num) +".jpg", filter_img)
    save_sequence_num+=1
    print ("[%s]   is processed." % (save_sequence_num))
    
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()