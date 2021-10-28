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

#read_start = 0
Padding_H  = -5

#Padding_H  = 254
#from  path_finding import PATH
Display_STD_flag = False
Padd_zero_top = True
Display_signal_flag = False
Display_Matrix_flag = True
save_matlab_flag = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
video_sizeH= 450
video_sizeW= 900

#videoout = cv2.VideoWriter(save_display_dir+'0output.avi', -1, 20.0, (video_sizeW,video_sizeH))

class Derivation_validate(object):
    def  __init__(self, H,W):
        self.Len_steam = 3
        self.crop_startH = 0
        self.cropH = int(800)
        #self.cropH = 700

        # the sheath std
        #self.crop_startH = 1
        #self.cropH = int(230)
        

        self.crop_startW = 100
        self.crop_startW = 1

        #self.crop_startW = 360

        self.cropW = int(800)
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
        self.currenta=0
        self.currentb =0
        self.max_switch = 0
    def  buffer(self,img1,img2):
        self.cnt +=1
        if self.initial ==0:
           mask = (img1 > 100)
        if self.cnt%self.sample_rate==0:
            #img1= np.clip(img1,85,255)-85
            #img2= np.clip(img2,85,255)-85
            #img1  = (img1 > 85)*  (img1*0+254)
            #img2  = (img2 > 85) * (img2*0+254)
            #img1  = (img1 > 85)*  img1
            #img2  = (img2 > 85) * img2 
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

        L,H,W = a_stack.size() 

        #stda =  a_stack.std( dim = 0)
        avg = torch.sum(a_stack,dim=0) / L
        dev  =  a_stack - avg
        dev2 = torch.abs (dev)
        stda =  torch.sum(dev2,dim=0) / L

        #mask1 = (stda > 20)
        mask1 = (stda > 20)

        mask2 = (avg  > 30)
        mask  = mask2 * mask1
        stda = stda*mask
        stda =  (torch.sum(mask)+1)
        #stda = float(stda.data.mean())

        L,H,W = b_stack.size() 

        #stda =  a_stack.std( dim = 0)
        avg = torch.sum(b_stack,dim=0) / L
 
        dev  =  b_stack - avg
        dev2 = torch.abs (dev)

        stdb =  torch.sum(dev2,dim=0) / L
        #mask1 = (stdb > 20)
        mask1 = (stdb > 20)

 
        mask2 = (avg  > 30)
        mask  = mask2 * mask1
        stdb = stdb*mask
        stdb = (torch.sum(mask)+1)
        #stdb=  b_stack.std( dim = 0)
        #stdb = float( stdb.data.mean())
        self.max_switch = 0
        if self.cnt>self.Len_steam:
            if (stda>self.maxa):
                self.maxa = stda
            if (stdb>self.maxb):
                    self.max_switch = 1
                    self.maxb = stdb
            self.std_suma += stda
            self.std_sumb += stdb
            self.currenta  = stda
            self.currentb  = stdb
            self.avga=self.std_suma /(self.cnt-self.Len_steam)
            self.avgb=self.std_sumb /(self.cnt-self.Len_steam)
  




        return stda, stdb

    pass
    def calculate2(self):
        a_stack  =  torch.from_numpy(self.steam1)
        b_stack  =  torch.from_numpy(self.steam2)
        #a_stack=a_stack.sum(dim=1)/(self.cropH -self.crop_startH)
        #b_stack=b_stack.sum(dim=1)/(self.cropH -self.crop_startH)

        L,H,W = a_stack.size() 

        #stda =  a_stack.std( dim = 0)
        avg = torch.sum(a_stack,dim=0) / L
        dev  =  a_stack[0,:,:] - a_stack[1,:,:]
        dev2 = torch.abs (dev)
        stda = dev2 

        #mask1 = (stda > 20)
        mask1 = (stda >60)

        mask2 = (avg  > 30)
        mask  = mask2 * mask1
        stda = stda*mask
        stda =  (torch.sum(mask)+1)
        #stda = float(stda.data.mean())

        L,H,W = b_stack.size() 

        #stda =  a_stack.std( dim = 0)
        avg = torch.sum(b_stack,dim=0) / L
        dev  =  b_stack[0,:,:] - b_stack[1,:,:]
        dev2 = torch.abs (dev)
        stdb = dev2 
        #mask1 = (stdb > 20)
        mask1 = (stdb > 60)

 
        mask2 = (avg  > 30)
        mask  = mask2 * mask1
        stdb = stdb*mask
        stdb = (torch.sum(mask)+1)
        #stdb=  b_stack.std( dim = 0)
        #stdb = float( stdb.data.mean())
        if self.cnt>self.Len_steam:
            if (stda>self.maxa):

                self.maxa = stda
            if (stdb>self.maxb):
                    self.maxb = stdb
            self.std_suma += stda
            self.std_sumb += stdb
            self.currenta  = stda
            self.currentb  = stdb
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
    if Padd_zero_top ==True:
            if Padding_H>0:
                padding = np.zeros((Padding_H,W_ini))
                img  = np.append(padding,img,axis=0)
            else:
                img = img[-Padding_H:H,:]
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
            img_path1 = savedir_process + str(i+5)+ ".jpg"
            video1 = cv2.imread(img_path1)
            gray_video1  =   cv2.cvtColor(video1, cv2.COLOR_BGR2GRAY)
            gray_video1 = gray_video1  
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
            img_path2 = operatedir_video + str(i+5)+ ".jpg"
            video2 = cv2.imread(img_path2)  
            gray_video2  =   cv2.cvtColor(video2, cv2.COLOR_BGR2GRAY)
            gray_video2 = gray_video2    
            gray_video2 = cv2.resize(gray_video2, (W_ini,H_ini), interpolation=cv2.INTER_AREA)
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
            if Display_Matrix_flag == True:
                img_path3 = operatedir_matrix + str(i+5)+ ".jpg"
                MATRIX_RESULT = cv2.imread(img_path3)
                MATRIX_RESULT  =   cv2.cvtColor(MATRIX_RESULT, cv2.COLOR_BGR2GRAY)
                Rotate_matr = cv2.rotate(MATRIX_RESULT,rotateCode = 2) 
                
                #show_2  = np.append(circular[:,300:W-300],Rotate_matr,axis=1) # cascade
                #show_2 = np.append(show_2,gray_video1[:,300:W-300],axis=1) # cascade
                #zero = np.zeros ((circular1.shape[0],10))
                Rotate_matr = cv2.resize (Rotate_matr, (71,circular1.shape[0]), interpolation=cv2.INTER_AREA)
                show_2  = np.append(circular[:,:],Rotate_matr,axis=1) # cascade
                show_2  = np.append(show_2,circular1[:,:],axis=1) # cascade

                #zero = np.zeros ((rectan2.shape[0],50))

                #show_2  = np.append(rectan2[:,:],zero,axis=1) # cascade
                #show_2  = np.append(show_2,rectan1[:,:],axis=1) # cascade

                #show_2 = cv2.resize(show_2, (int(show_2.shape[1]/1.5),int(show_2.shape[0]/1.5)), interpolation=cv2.INTER_AREA)
                show_2 = cv2.resize(show_2, (int(show_2.shape[1]/1.1),int(show_2.shape[0]/1.1)), interpolation=cv2.INTER_AREA)
            #cv2.imshow('matrix',MATRIX_RESULT)
            else: 
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


            if(i == read_start): # initialize the color sequence 
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
            if Display_STD_flag  ==True :
                STD_call.buffer(rectan1 ,rectan2 )
                std1,std2=STD_call.calculate()
                if (STD_call.max_switch ==1):
                    print("new max appear!")
                print("correct:"+str(std1)) # 1 is the process
                print("origin:"+str(std2))
                print("correct_ave:"+str(STD_call.avga))
                print("origin_ave:"+str(STD_call.avgb))
                print("correct_max:"+str(STD_call.maxa))
                print("origin_max:"+str(STD_call.maxb))
                
                if save_matlab_flag == True:
                    matlab_saver.buffer2(std1,std2)
                pass

            print("update"+str(i)+":")

            if cv2.waitKey(1) & 0xFF == ord('q'):
              break

    pass
####################################################
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
                std1,std2=STD_call.calculate2()
                print("correct:"+str(std1))
                print("origin:"+str(std2))
                print("correct_ave:"+str(STD_call.avga))
                print("origin_ave:"+str(STD_call.avgb))
                if save_matlab_flag == True:
                    matlab_saver.buffer2(std1,std2)

                
                pass

            print("update"+str(i)+":")

            if cv2.waitKey(1) & 0xFF == ord('q'):
              break

    pass

if __name__ == '__main__':
    diplay_sequence()
    #displayselected()
