root =  "E:/database/NURD/8th 10 2021 colection for MedIA/"
root =  "E:/database/NURD/3rd December Tube/"
root =  "E:/database/Polyp detection/"

#root = "E:/database/NURD/20th October/"
operatedir =   root + "raw/disturb_p6/"
operatedir =   root + "raw/phantom_pull_0.1_dis3/"
operatedir =   root + "raw/tube1/"
operatedir =   root + "raw/tube1slow/"
operatedir =   root + "raw/tube2 slow long/"
operatedir =   root + "raw/tube3vain/"
operatedir =   root + "raw/sqr2/"
operatedir =   root + "raw/video1/"



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
# from Correct_sequence_integral import read_start
from read_circu import tranfer_frome_rec2cir
from basic_trans import Basic_oper
from  matlab import Save_Signal_matlab
from display import tranfer2circ_padding as rec_2_cir

base_dir =  os.path.basename(os.path.normpath(operatedir))
save_dir =  root + "resize/"  + base_dir + "/"
save_dir_cir =   root + "resize_circular/"  + base_dir + "/"
try:
    os.stat(save_dir)
except:
    os.makedirs(save_dir)

try:
    os.stat(save_dir_cir)
except:
    os.makedirs(save_dir_cir)
read_start1 = 79
read_start2 = 79

Padding_H  = 0

#Padding_H  = 254
#from  path_finding import PATH
Display_STD_flag = False
Padd_zero_top = True
Display_signal_flag = False
Display_Matrix_flag = False
save_matlab_flag = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
video_sizeH= 450
video_sizeW= 900

#videoout = cv2.VideoWriter(save_display_dir+'0output.avi', -1, 20.0, (video_sizeW,video_sizeH))
 
if Display_signal_flag == True:
    from analy import MY_ANALYSIS
    #show the stastics results
    saved_stastics  = MY_ANALYSIS()
    saved_stastics=saved_stastics.read_my_signal_results()
    saved_stastics.display()
def tranfer2circ_padding(img):
    H,W_ini = img.shape
    padding = np.zeros((Padding_H,W_ini))
    if Padd_zero_top ==True:
            img  = np.append(padding,img,axis=0)
    circular = tranfer_frome_rec2cir(img)
    return circular

def diplay_sequence():
    
    #show the image results
    read_sequence = os.listdir(operatedir)
    

    this_id  = 1
    for dir in read_sequence:
    #for i in os.listdir("E:/estimagine/vs_project/PythonApplication_data_au/pic/"):
    ##      process
            img_path1 = operatedir +  dir
            video1 = cv2.imread(img_path1)

            grayr=cv2.rotate(video1,rotateCode = cv2.ROTATE_90_CLOCKWISE) 
            gray_cir  =   cv2.cvtColor(grayr, cv2.COLOR_BGR2GRAY)
            H,W = gray_cir.shape

            #grayr = cv2.resize(gray_cir, (832,H), interpolation=cv2.INTER_LINEAR)

            gray_cir = cv2.resize(gray_cir, (832,832), interpolation=cv2.INTER_LINEAR)
            gray_cir =  rec_2_cir(gray_cir)
            cv2.imwrite(save_dir  + str(this_id) +".jpg",grayr )
            cv2.imwrite(save_dir_cir  + str(this_id) +".jpg",gray_cir )

            this_id = this_id +1
            print(dir)
    pass
####################################################
 

if __name__ == '__main__':
    diplay_sequence()
    #displayselected()
