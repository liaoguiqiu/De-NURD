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
#read_start = 1500
Padding_H  = 1
#from  path_finding import PATH
Display_STD_flag = False
Padd_zero_top = False
Display_signal_flag = False
Display_Matrix_flag = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Derivation_validate(object):
    def  __init__(self, H,W):
        Len_steam = 10
        self.crop_start = 100
        self.cropH = int(H/2)
        self.steam1=np.zeros((Len_steam,self.cropH-self.crop_start,W))
        self.steam2=np.zeros((Len_steam, self.cropH-self.crop_start,W))
        self.cnt=0
        self.sample_rate=5
        self.std_suma =0
        self.std_sumb =0
        self.avga =0
        self.avgb =0

    def  buffer(self,img1,img2):
        self.cnt +=1
        if self.cnt%self.sample_rate==0:
            self.steam1=np.append(self.steam1, [img1[self.crop_start:self.cropH,:]] ,axis=0) # save sequence
            # no longer delete the fist  one
            self.steam2=np.append(self. steam2, [img2[self.crop_start:self.cropH,:]] ,axis=0) # save sequence
            # no longer delete the fist  one
        
            self.steam2= np.delete(self.steam2 , 0,axis=0)
            self.steam1= np.delete(self.steam1 , 0,axis=0)
        pass
    def calculate(self):
        a_stack  =  torch.from_numpy(self.steam1)
        b_stack  =  torch.from_numpy(self.steam2)
        stda =  a_stack.std( dim = 0)
        stda = float(stda.data.mean())
        stdb=  b_stack.std( dim = 0)
        stdb = float( stdb.data.mean())
        self.std_suma += stda
        self.std_sumb += stdb
        self.avga=self.std_suma /self.cnt
        self.avgb=self.std_sumb /self.cnt




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
            img_path1 = savedir_process + str(i+20)+ ".jpg"
            video1 = cv2.imread(img_path1)
            gray_video1  =   cv2.cvtColor(video1, cv2.COLOR_BGR2GRAY)
            rectan1 = gray_video1
            circular1 = tranfer2circ_padding(gray_video1)
            gray_video1 = circular1
            #cv2.imshow('circular video',circular )      
            # processe
            #img_path1 = savedir_path + str(i+10)+ ".jpg"
            #video1 = cv2.imread(img_path1)
            #gray_video1  =   cv2.cvtColor(video1, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('step_process',gray_video1)  

            # raws 
            img_path2 = operatedir_video + str(i+20)+ ".jpg"
            video2 = cv2.imread(img_path2)
            gray_video2  =   cv2.cvtColor(video2, cv2.COLOR_BGR2GRAY)
            gray_video2 = cv2.resize(gray_video2, (W_ini,H_ini), interpolation=cv2.INTER_AREA)
            rectan2 = gray_video2
            circular= tranfer2circ_padding(gray_video2)
            #cv2.imshow('circular video',circular ) 

            # matrix
        
            if Display_Matrix_flag == True:
                img_path3 = operatedir_matrix + str(i+10)+ ".jpg"
                MATRIX_RESULT = cv2.imread(img_path3)
                MATRIX_RESULT  =   cv2.cvtColor(MATRIX_RESULT, cv2.COLOR_BGR2GRAY)
                Rotate_matr = cv2.rotate(MATRIX_RESULT,rotateCode = 2) 
                show_2  = np.append(circular[:,300:W-300],Rotate_matr,axis=1) # cascade
                show_2 = np.append(show_2,gray_video1[:,300:W-300],axis=1) # cascade
            #cv2.imshow('matrix',MATRIX_RESULT)
            else: 
                #show_2  = np.append(circular[:,300:W_ini-300],gray_video1[:,300:W_ini-300],axis=1) # cascade
                #show_2  = np.append(circular[:,:],gray_video1[:,:],axis=1) # cascade
                zero = np.zeros ((rectan2.shape[0],50))
                #show_2  = np.append(circular,zero,axis=1) # cascade
                #show_2  = np.append(show_2,gray_video1,axis=1) # cascade
                show_2  = np.append(rectan2[:,:],zero,axis=1) # cascade
                show_2  = np.append(show_2,rectan1[:,:],axis=1) # cascade



            if(i == read_start): # initialize the color sequence 
                stream=np.zeros((show_2.shape[0],show_2.shape[1],3))
            else:

                if i%1==0:
                    new_frame   = np.zeros((show_2.shape[0],show_2.shape[1],1))
                    new_frame[:,:,0]  = show_2
                    stream=np.append(stream,new_frame,axis=2) # save sequence
                    stream= np.delete(stream , 0,axis=2) # update this every 50 frame
 

            cv2.imshow('combin video',show_2 ) 
            cv2.imshow('show 3 imgs sequence with color',stream.astype(np.uint8) ) 
            cv2.imwrite(save_display_dir  + str(i) +".jpg",stream )
            cv2.imwrite(savedir_origin_circle  + str(i) +".jpg",circular )
            cv2.imwrite(savedir_process_circle  + str(i) +".jpg",gray_video1 )
            if Display_STD_flag  ==True :
                STD_call.buffer(rectan1 ,rectan2 )
                std1,std2=STD_call.calculate()
                print("correct:"+str(std1))
                print("origin:"+str(std2))
                print("correct_ave:"+str(STD_call.avga))
                print("origin_ave:"+str(STD_call.avgb))

                
                pass

            print("update"+str(i)+":")

            if cv2.waitKey(1) & 0xFF == ord('q'):
              break

    pass

def displayselected():
    
    #show the image results
    read_sequence = os.listdir(savedir_process)
    seqence_Len = len(read_sequence)
    img_path1 = savedir_process + str(read_start)+ ".jpg"
    video2 = cv2.imread(img_path1)
    gray_video2  =   cv2.cvtColor(video2, cv2.COLOR_BGR2GRAY)
    H_ini,W_ini= gray_video2.shape
    STD_call  = Derivation_validate(H_ini,W_ini)

    iteration =0
    for i in  range(read_start ,read_start+800,50):
    #for i in os.listdir("E:/estimagine/vs_project/PythonApplication_data_au/pic/"):
    ##      process

            img_path1 = savedir_process + str(i+20)+ ".jpg"
            video1 = cv2.imread(img_path1)
            gray_video1  =   cv2.cvtColor(video1, cv2.COLOR_BGR2GRAY)
            rectan1 = gray_video1
            circular1 = tranfer2circ_padding(gray_video1)
            gray_video1 = circular1
            #cv2.imshow('circular video',circular )      
            # processe
            #img_path1 = savedir_path + str(i+10)+ ".jpg"
            #video1 = cv2.imread(img_path1)
            #gray_video1  =   cv2.cvtColor(video1, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('step_process',gray_video1)  

            # raws 
            img_path2 = operatedir_video + str(i+20)+ ".jpg"
            video2 = cv2.imread(img_path2)
            gray_video2  =   cv2.cvtColor(video2, cv2.COLOR_BGR2GRAY)
            gray_video2 = cv2.resize(gray_video2, (W_ini,H_ini), interpolation=cv2.INTER_AREA)
            rectan2 = gray_video2
            circular= tranfer2circ_padding(gray_video2)
            #cv2.imshow('circular video',circular ) 

            # matrix
        
            if Display_Matrix_flag == True:
                img_path3 = operatedir_matrix + str(i+10)+ ".jpg"
                MATRIX_RESULT = cv2.imread(img_path3)
                MATRIX_RESULT  =   cv2.cvtColor(MATRIX_RESULT, cv2.COLOR_BGR2GRAY)
                Rotate_matr = cv2.rotate(MATRIX_RESULT,rotateCode = 2) 
                show_2  = np.append(circular[:,300:W-300],Rotate_matr,axis=1) # cascade
                show_2 = np.append(show_2,gray_video1[:,300:W-300],axis=1) # cascade
            #cv2.imshow('matrix',MATRIX_RESULT)
            else: 
                #show_2  = np.append(circular[:,300:W_ini-300],gray_video1[:,300:W_ini-300],axis=1) # cascade
                show_2  = np.append(rectan2[:,:],rectan1[:,:],axis=1) # cascade


            if(i == read_start): # initialize the color sequence 
                stream=np.zeros((show_2.shape[0],show_2.shape[1],3))
                new_frame   = np.zeros((show_2.shape[0],show_2.shape[1],1))
                new_frame[:,:,0]  = show_2
                stream=np.append(stream,new_frame,axis=2) # save sequence
                
                stream= np.delete(stream , 0,axis=2) # update this every 50 frame
                 
            else:
                new_frame   = np.zeros((show_2.shape[0],show_2.shape[1],1))
                new_frame[:,:,0]  = show_2
                stream=np.append(stream,new_frame,axis=2) # save sequence
                stream= np.delete(stream , 0,axis=2) # update this every 50 frame
                #if iteration< 3:
                #    stream= np.delete(stream , 0,axis=2) # update this every 50 frame
                #else:
                #    stream= np.delete(stream , 1,axis=2)
            
            iteration+=1
            cv2.imshow('combin video',show_2 ) 
            cv2.imshow('show 3 imgs sequence with color',stream.astype(np.uint8) ) 
            cv2.imwrite(save_display_dir  + str(1) +".jpg",stream )
            cv2.imwrite(savedir_origin_circle  + str(1) +".jpg",circular )
            cv2.imwrite(savedir_process_circle  + str(1) +".jpg",gray_video1 )
            if Display_STD_flag  ==True :
                STD_call.buffer(rectan1 ,rectan2 )
                std1,std2=STD_call.calculate()
                print("correct:"+str(std1))
                print("origin:"+str(std2))
                print("correct_ave:"+str(STD_call.avga))
                print("origin_ave:"+str(STD_call.avgb))


                
                pass

            print("update"+str(i)+":")

            if cv2.waitKey(1) & 0xFF == ord('q'):
              break

    pass

if __name__ == '__main__':
    diplay_sequence()
    #displayselected()
