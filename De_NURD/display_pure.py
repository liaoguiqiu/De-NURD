savedir_process = "../../saved_processed_polar/"
operatedir_video = "../../saved_original/"

#savedir_process = "../../saved_pair2/"
#operatedir_video = "../../saved_pair1/"

operatedir_matrix  =  "../../saved_matrix/"
 
operatedir_one =  "../../saved_matrix/126.jpg"
#operatedir_video = "E:/PhD/trying/saved_filtered_img/"
savedir_path = "../../saved_processed/"
savedir_process_circle = "../../saved_processed/"

#saved_original_circular
savedir_origin_circle =  "../../saved_original_circular/"
#saved_processed_polar
save_display_dir = "../../saved_display_compare/"
save_displaygray_dir = "../../saved_display_compare_gray/"



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
from Correct_sequence_integral import read_start 
from read_circu import tranfer_frome_rec2cir
from basic_trans import Basic_oper
from  matlab import Save_Signal_matlab
matlab_saver  = Save_Signal_matlab()

#read_start = 1500
Padding_H  = 0

#Padding_H  = 0
#from  path_finding import PATH
Display_STD_flag = False
Padd_zero_top = True
Display_signal_flag = False
Display_Matrix_flag = False
save_matlab_flag = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Derivation_validate(object):
    def  __init__(self, H,W):
        self.Len_steam = 3
        self.crop_startH = 000
        self.cropH = int(1000)
        self.crop_startW = 200
        #self.crop_startW = 360

        self.cropW = int(500)
        #self.crop_startH = 0
        #self.cropH = int(700)
        #self.crop_startW = 0
        #self.cropW = int(720)
        self.steam1=np.zeros((self.Len_steam,self.cropH-self.crop_startH,self.cropW-self.crop_startW))
        self.steam2=np.zeros((self.Len_steam, self.cropH-self.crop_startH,self.cropW-self.crop_startW))
        self.cnt=0
        self.sample_rate=1
        self.std_suma =0
        self.std_sumb =0
        self.avga =0
        self.avgb =0
        self.initial=0
        self.vara=0
        self.varb =0
        self.maxa=0
        self.maxb =0

    def  buffer(self,img1,img2):
        self.cnt +=1
        if self.initial ==0:
           mask = (img1 > 100)
        if self.cnt%self.sample_rate==0:
            #img1= np.clip(img1,85,255)-85
            #img2= np.clip(img2,85,255)-85
            #img1  = (img1 > 85)*  (img1*0+254)
            #img2  = (img2 > 85) * (img2*0 +254)
            img1  = (img1 > 85)*  img1
            img2  = (img2 > 85) * img2 
            #source1  =  cv2.GaussianBlur(img1,(5,5),0)
            #source2  =  cv2.GaussianBlur(img2,(5,5),0)
            source1  =  img1 
            source2  =  img2 


            self.steam1=np.append(self.steam1, [source1[self.crop_startH:self.cropH,
                                                        self.crop_startW:self.cropW]] ,axis=0) # save sequence
            # no longer delete the fist  one
            self.steam2=np.append(self. steam2, [source2[self.crop_startH:self.cropH,
                                                         self.crop_startW:self.cropW]] ,axis=0) # save sequence
            # no longer delete the fist  one
        
            self.steam2= np.delete(self.steam2 , 0,axis=0)
            self.steam1= np.delete(self.steam1 , 0,axis=0)
        pass
    def calculate(self):
        a_stack  =  torch.from_numpy(self.steam1)
        b_stack  =  torch.from_numpy(self.steam2)
        #a_stack=a_stack.sum(dim=1)/(self.cropH -self.crop_startH)
        #b_stack=b_stack.sum(dim=1)/(self.cropH -self.crop_startH)


        stda =  a_stack.std( dim = 0)
        stda = float(stda.data.mean())
        stdb=  b_stack.std( dim = 0)
        stdb = float( stdb.data.mean())
        if self.cnt>self.Len_steam:
            if (stda>self.maxa):
                self.maxa = stda
            if (stdb>self.maxb):
                    self.maxb = stdb
            self.std_suma += stda
            self.std_sumb += stdb

            self.avga=self.std_suma /(self.cnt-self.Len_steam)
            self.avgb=self.std_sumb /(self.cnt-self.Len_steam)
  




        return stda, stdb

    pass


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
    read_sequence = os.listdir(savedir_process)
    seqence_Len = len(read_sequence)
    img_path1 = savedir_process + str(read_start+20)+ ".jpg"
    video2 = cv2.imread(img_path1)
    gray_video2  =   cv2.cvtColor(video2, cv2.COLOR_BGR2GRAY)
    H_ini,W_ini= gray_video2.shape
    STD_call  = Derivation_validate(H_ini,W_ini)


    for i in range(read_start,seqence_Len+read_start):
    #for i in os.listdir("E:/estimagine/vs_project/PythonApplication_data_au/pic/"):
    ##      process
            img_path1 = save_displaygray_dir + str(i+20)+ ".jpg"
            video1 = cv2.imread(img_path1)
            

            # raws 
            img_path2 = save_display_dir + str(i+20)+ ".jpg"
            video2 = cv2.imread(img_path2)
             
             

            cv2.imshow('combin video',video1.astype(np.uint8)) 
            cv2.imshow('show 3 imgs sequence with color',video2.astype(np.uint8)) 
           

            print("update"+str(i)+":")

            if cv2.waitKey(1) & 0xFF == ord('q'):
              break

    pass
####################################################
 

if __name__ == '__main__':
    diplay_sequence()
    #displayselected()
