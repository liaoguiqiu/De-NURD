operatedir_matrix  =  "..\\..\\saved_matrix\\"
 
operatedir_one =  "..\\..\\saved_matrix\\126.jpg"
operatedir_video = "..\\..\\saved_original\\"
#operatedir_video = "E:\\PhD\\trying\\saved_filtered_img\\"
savedir_path = "..\\..\\saved_processed\\"
save_display_dir = "..\\..\\saved_display_compare\\"


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

Display_signal_flag = False
Display_Matrix_flag = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if Display_signal_flag == True:
    from analy import MY_ANALYSIS
    #show the stastics results
    saved_stastics  = MY_ANALYSIS()
    saved_stastics=saved_stastics.read_my_signal_results()
    saved_stastics.display()


#show the image results
read_sequence = os.listdir(savedir_path)
seqence_Len = len(read_sequence)



for i in range(seqence_Len):
#for i in os.listdir("E:\\estimagine\\vs_project\\PythonApplication_data_au\\pic\\"):     
        # processe
        img_path1 = savedir_path + str(i+10)+ ".jpg"
        video1 = cv2.imread(img_path1)
        gray_video1  =   cv2.cvtColor(video1, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('step_process',gray_video1)  

        # raws 
        img_path2 = operatedir_video + str(i+20)+ ".jpg"
        video2 = cv2.imread(img_path2)
        gray_video2  =   cv2.cvtColor(video2, cv2.COLOR_BGR2GRAY)
         
        new_frame2=cv2.rotate(gray_video2,rotateCode = 2) 
        new_frame3= new_frame2.astype(float)
        H,W= new_frame2.shape
        circular3=np.ones((H,W))
        circular2=circular3.astype(float)
        circular = circular2*2
        circular = cv2.linearPolar(new_frame3, (int(W/2) , int(H/2)),200, cv2.WARP_INVERSE_MAP)
        circular=circular.astype(np.uint8)
        #cv2.imshow('circular video',circular ) 

        # matrix
        img_path3 = operatedir_matrix + str(i+10)+ ".jpg"
        MATRIX_RESULT = cv2.imread(img_path3)
        MATRIX_RESULT  =   cv2.cvtColor(MATRIX_RESULT, cv2.COLOR_BGR2GRAY)
        if Display_Matrix_flag == True:
            Rotate_matr = cv2.rotate(MATRIX_RESULT,rotateCode = 2) 
            show_2  = np.append(circular[:,300:W-300],Rotate_matr,axis=1) # cascade
            show_2 = np.append(show_2,gray_video1[:,300:W-300],axis=1) # cascade
        #cv2.imshow('matrix',MATRIX_RESULT) 
        show_2  = np.append(circular[:,300:W-300],gray_video1[:,300:W-300],axis=1) # cascade

        if(i == 0): # initialize the color sequence 
            stream=np.zeros((show_2.shape[0],show_2.shape[1],3))
        else:
            new_frame   = np.zeros((show_2.shape[0],show_2.shape[1],1))
            new_frame[:,:,0]  = show_2
            stream=np.append(stream,new_frame,axis=2) # save sequence
            stream= np.delete(stream , 0,axis=2)
            

        cv2.imshow('combin video',show_2 ) 
        cv2.imshow('show 3 imgs sequence with color',stream.astype(np.uint8) ) 
        cv2.imwrite(save_display_dir  + str(i) +".jpg",stream )

        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
