#operatedir_video = "D:/PhD/trying/tradition_method/OCT/sheath registration/pair A/with rectan/"
#operatedir_video = "D:/PhD/trying/tradition_method/OCT/sheath registration/pairC/rectangular/2/"
#operatedir_video = "E:/database/Needle injection/3D scan/0/"
operatedir_video =   "E:/database/Needle injection/28th Jan/raw_pullback/1/"
operatedir_video2 =  "E:/database/Needle injection/28th Jan/raw_pullback/1/"
#operatedir_video2 = "D:/PhD/trying/tradition_method/OCT//sheath registration/pairD/phantom/1/"




#operatedir_video2 = "D:/PhD/trying/tradition_method/OCT/sheath registration/pair A/without rectan/"
#operatedir_video2 = "D:/PhD/trying/tradition_method/OCT//sheath registration/pairC/rectangular/2/"
#operatedir_video2 = "D:/PhD/trying/tradition_method/OCT//sheath registration/pairD/ref/"
#operatedir_video2 = "D:/PhD/trying/tradition_method/OCT//sheath registration/pairD/phantom/1/"|


##operatedir_video2 = "D:/PhD/trying/tradition_method/OCT/sheath registration/pairC/ruler/2/"
#operatedir_video2 = "D:/PhD/trying/tradition_method/OCT/sheath registration/pairC/phantom/1/"


#operatedir_video2 = "D:/PhD/trying/tradition_method/OCT/sheath registration/pairC/ruler/2/"
#operatedir_video2 = "D:/PhD/trying/tradition_method/OCT/sheath registration/pairC/rectangular/3/"

#save_dir_rectan  =  "D:/PhD/trying/tradition_method/OCT/sheath registration/pairC/ruler/2_/"
#save_dir_rectan  =  "../../saved_original/"
save_dir_rectan =  "E:/database/Needle injection/28th Jan/pullback/1/polar/"
savedir_origin_circle =  "E:/database/Needle injection/28th Jan/pullback/1/cartesian/"


#savedir_process = "../../saved_pair2/"
#operatedir_video = "../../saved_pair1/"
 
#operatedir_video = "E:/PhD/trying/saved_filtered_img/"
savedir_path = "../../saved_processed/"
savedir_process_circle = "../../saved_processed/"

#saved_original_circular
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
    read_sequence = os.listdir(operatedir_video)
    seqence_Len = len(read_sequence)
    img_path1 = operatedir_video + "image"+ str(read_start1)+ ".jpg"
    video2 = cv2.imread(img_path1)
    gray_video2  =   cv2.cvtColor(video2, cv2.COLOR_BGR2GRAY)
    gray_video2 = cv2.resize(gray_video2, (1000,1000), interpolation=cv2.INTER_AREA)

    H_ini,W_ini= gray_video2.shape
    

    First  = 0
    for i in range(read_start1,seqence_Len ):
    #for i in os.listdir("E:/estimagine/vs_project/PythonApplication_data_au/pic/"):
    ##      process
            img_path1 = operatedir_video +  "image" +  str(i  )+ ".jpg"
            video1 = cv2.imread(img_path1)
            gray_video1  =   cv2.cvtColor(video1, cv2.COLOR_BGR2GRAY)
            gray_video1 =      cv2.resize(gray_video1, (W_ini,H_ini), interpolation=cv2.INTER_AREA) 
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
            img_path2 = operatedir_video2 +  "image" +  str(i+read_start2-read_start1)+ ".jpg"
            #img_path2 = operatedir_video2   +  str(i+read_start2)+ ".jpg"

            video2 = cv2.imread(img_path2)
            cv2.imwrite(save_dir_rectan  + str(i) +".jpg",video2 ) 
            gray_video2  =   cv2.cvtColor(video2, cv2.COLOR_BGR2GRAY)
            gray_video2 =   cv2.resize(gray_video2, (W_ini,H_ini), interpolation=cv2.INTER_AREA)  

             
            rectan2 = gray_video2 
            circular= tranfer2circ_padding(gray_video2)
            #cv2.imshow('circular video',circular ) 

            # matrix
            #circular  = (circular > 70) #* (circular*0+200)
            #circular1  = (circular1 > 70) #* (circular1*0+200)
            #circular  = (circular > 70) * circular
            #circular1  = (circular1 > 70) * circular1
            #rectan1  = (rectan1 > 111) * rectan1
            #rectan2  = (rectan2 > 111) * rectan2
             
                #show_2  = np.append(circular[:,300:W-300],Rotate_matr,axis=1) # cascade
            
                #show_2  = np.append(circular[:,300:W_ini-300],gray_video1[:,300:W_ini-300],axis=1) # cascade
                #show_2  = np.append(circular[:,:],gray_video1[:,:],axis=1) # cascade
                #show_2  = np.append(circular,zero,axis=1) # cascade
                #show_2  = np.append(show_2,gray_video1,axis=1) # cascade
                #show_2  = np.append(rectan2[:,:],zero,axis=1) # cascade
                #show_2  = np.append(show_2,rectan1[:,:],axis=1) # cascade
            zero = np.zeros ((circular1.shape[0],10))

            show_2  = np.append(circular[:,:],zero,axis=1) # cascade
            show_2  = np.append(show_2,circular1[:,:],axis=1) # cascade

            #zero = np.zeros ((rectan2.shape[0],50))

            #show_2  = np.append(rectan2[:,:],zero,axis=1) # cascade
            #show_2  = np.append(show_2,rectan1[:,:],axis=1) # cascade

            #show_2 = cv2.resize(show_2, (int(show_2.shape[1]/1.5),int(show_2.shape[0]/1.5)), interpolation=cv2.INTER_AREA)
            show_2 = cv2.resize(show_2, (int(show_2.shape[1]/1.1),int(show_2.shape[0]/1.1)), interpolation=cv2.INTER_AREA)

                #show_2 = cv2.resize(show_2, (int(video_sizeW),int(video_sizeH)), interpolation=cv2.INTER_AREA)


            if(First == 0): # initialize the color sequence 
                First  = 1 
                stream=np.zeros((show_2.shape[0],show_2.shape[1],3))
            else:

                if i%1==0:
                    new_frame   = np.zeros((show_2.shape[0],show_2.shape[1],1))
                    new_frame[:,:,0]  = show_2
                    stream=np.append(stream,new_frame,axis=2) # save sequence
                    stream= np.delete(stream , 0,axis=2) # update this every 50 frame
 

            #cv2.imshow('combin video',show_2.astype(np.uint8)) 
            cv2.imshow('show 3 imgs sequence with color',stream.astype(np.uint8) ) 
            #videoout.write(stream)
            cv2.imwrite(save_display_dir  + str(i) +".jpg",stream )
            cv2.imwrite(save_displaygray_dir  + str(i) +".jpg",show_2 )

            cv2.imwrite(savedir_origin_circle  + str(i) +".jpg",circular )
            cv2.imwrite(savedir_process_circle  + str(i) +".jpg",gray_video1 )
            

            print("update"+str(i)+":")

            if cv2.waitKey(1) & 0xFF == ord('q'):
              break

    pass
####################################################
 

if __name__ == '__main__':
    diplay_sequence()
    #displayselected()
