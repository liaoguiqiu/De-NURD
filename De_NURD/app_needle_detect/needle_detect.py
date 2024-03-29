savedir_process = "E:/database/Needle injection/28th Jan/back_forward/1/cartesian/"
operatedir_video = "E:/database/Needle injection/28th Jan/back_forward/1/cartesian/"
out_dir = "E:/database/Needle injection/28th Jan/back_forward/1/output/"


#savedir_process = "../../saved_pair2/"
#operatedir_video = "../../saved_pair1/"
 
 
operatedir_one =  "../../saved_matrix/126.jpg"
#operatedir_video = "E:/PhD/trying/saved_filtered_img/"
savedir_path = "../../saved_processed/"
savedir_process_circle = "../../saved_processed/"

#saved_original_circular
savedir_origin_circle =  "../../saved_original_circular/"
#saved_processed_polar
save_display_dir = "../../saved_display_compare/"
import time

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
from scipy import ndimage
import random
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
from Correct_sequence_integral import read_start 
from read_circu import tranfer_frome_rec2cir
from basic_trans import Basic_oper
from  matlab import Save_Signal_matlab
from line_detection import Line_detect
matlab_saver  = Save_Signal_matlab()

read_start = 0
read_end = 452

Padding_H  = 0

#Padding_H  = 254
#from  path_finding import PATH
Display_STD_flag = True
Padd_zero_top = True
Display_signal_flag = False
Display_Matrix_flag = False
save_matlab_flag = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
video_sizeH= 450
video_sizeW= 900

#videoout = cv2.VideoWriter(save_display_dir+'0output.avi', -1, 20.0, (video_sizeW,video_sizeH))

class Derivation_validate(object):
    def  __init__(self, H,W):
        self.Len_steam = 1
        self.crop_startH = 0
        self.cropH = H
        self.needledetector = Line_detect()
        

        
        self.crop_startW = 0

        #self.crop_startW = 360

        
        self.cropW = W

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

        L,H,W = a_stack.size() 

        #stda =  a_stack.std( dim = 0)
        avg = torch.sum(a_stack,dim=0) / L
        result_img=torch.Tensor.cpu(avg).detach().numpy()

        dev  =  a_stack - avg
        dev2 = torch.abs (dev)
        stda =  torch.sum(dev2,dim=0) / L
        final_result = cv2.cvtColor(result_img.astype(np.uint8),cv2.COLOR_GRAY2RGB)
        cv2.imshow('original',final_result.astype(np.uint8) ) 

        result =  self. needledetector.detection(result_img[150:650,150:650])
        final_result[150:650,150:650,:]=  result

        #result_img=torch.Tensor.cpu(stda).detach().numpy()
        return final_result
         

     
     

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
    img_path1 = savedir_process + str(read_start)+ ".jpg"
    video2 = cv2.imread(img_path1)
    gray_video2  =   cv2.cvtColor(video2, cv2.COLOR_BGR2GRAY)
    gray_video2 = cv2.resize(gray_video2, (800,800), interpolation=cv2.INTER_AREA)

    H_ini,W_ini= gray_video2.shape
    STD_call  = Derivation_validate(800,800)


    for i in range(read_start,read_end):
    #for i in os.listdir("E:/estimagine/vs_project/PythonApplication_data_au/pic/"):
    ##      process
            img_path1 = savedir_process + str(i)+ ".jpg"
            video1 = cv2.imread(img_path1)
            gray_video1  =   cv2.cvtColor(video1, cv2.COLOR_BGR2GRAY)
            gray_video1 = cv2.resize(gray_video1, (800,800), interpolation=cv2.INTER_AREA)

            gray_video1 = gray_video1  
            rectan1 = gray_video1
            #circular1 = tranfer2circ_padding(gray_video1)
            circular1 = gray_video1

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
            gray_video2 = gray_video2   
            gray_video2 = cv2.resize(gray_video2, (800,800), interpolation=cv2.INTER_AREA)
            rectan2 = gray_video2
            #circular= tranfer2circ_padding(gray_video2)
            circular= gray_video2

            #cv2.imshow('circular video',circular ) 

            # matrix
            #circular  = (circular > 70) #* (circular*0+200)
            #circular1  = (circular1 > 70) #* (circular1*0+200)
            #circular  = (circular > 70) * circular
            #circular1  = (circular1 > 70) * circular1
            #rectan1  = (rectan1 > 111) * rectan1
            #rectan2  = (rectan2 > 111) * rectan2
            if Display_Matrix_flag == True:
                img_path3 = operatedir_matrix + str(i+10)+ ".jpg"
                MATRIX_RESULT = cv2.imread(img_path3)
                MATRIX_RESULT  =   cv2.cvtColor(MATRIX_RESULT, cv2.COLOR_BGR2GRAY)
                Rotate_matr = cv2.rotate(MATRIX_RESULT,rotateCode = 2) 
                #show_2  = np.append(circular[:,300:W-300],Rotate_matr,axis=1) # cascade
                #show_2 = np.append(show_2,gray_video1[:,300:W-300],axis=1) # cascade
                
            #cv2.imshow('matrix',MATRIX_RESULT)
            else: 
              
                show_2 = circular

            if(i == read_start): # initialize the color sequence 
                stream=np.zeros((show_2.shape[0],show_2.shape[1],3))
            else:

                if i%1==0:
                    new_frame   = np.zeros((show_2.shape[0],show_2.shape[1],1))
                    new_frame[:,:,0]  = show_2
                    stream=np.append(stream,new_frame,axis=2) # save sequence
                    stream= np.delete(stream , 0,axis=2) # update this every 50 frame
 

            #cv2.imshow('combin video',show_2.astype(np.uint8)) 
            #cv2.imshow('show 3 imgs sequence with color',stream.astype(np.uint8) ) 
            #videoout.write(stream)
            cv2.imwrite(save_display_dir  + str(i) +".jpg",stream )
            cv2.imwrite(save_displaygray_dir  + str(i) +".jpg",show_2 )

            cv2.imwrite(savedir_origin_circle  + str(i) +".jpg",circular )
            cv2.imwrite(savedir_process_circle  + str(i) +".jpg",gray_video1 )
            if Display_STD_flag  ==True :
                 
                STD_call.buffer(rectan1 ,rectan2  )
                start_time  = time.time()

                final_result =  STD_call.calculate()
                end_time  = time.time()

                #final_result = ndimage.rotate(final_result, 45)
                cv2.imwrite(out_dir  + str(i) +".jpg",final_result )
                print (" all test point time is [%f] " % ( end_time - start_time))
                

            print("update"+str(i)+":")

            if cv2.waitKey(1) & 0xFF == ord('q'):
              break
    pass
if __name__ == '__main__':
    diplay_sequence()
    #displayselected()
