operatedir  =  "..\\saved_matrix\\8.jpg"
savedir_path = "..\\saved_processed\\"

import cv2
import math
import numpy as np
from median_filter_special import  myfilter

import os
import torch
import scipy.signal as signal
 
import random
from torch.autograd import Variable
from DeepPathsearch.dataset import myDataloader,Batch_size,Resample_size, Path_length
from DeepPathsearch import gan_body
from DeepPathsearch.image_trans import BaseTransform 
from scipy.ndimage import gaussian_filter1d
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from cost_matrix import Window_LEN 
dir_netD  = "..\\..\\DeepPathFinding\\out\\netD_epoch_50.pth"
transform = BaseTransform(  Resample_size,[104])
netD = gan_body._netD_Resnet()
print('load weights for Path_ find ing')
netD.load_state_dict(torch.load(dir_netD))
print(netD)
netD.cuda()
netD.eval()
class PATH:

    def search_a_path(img,start_p):

       
        img=img.astype(float)
         
        h, w= img.shape
        path = np.ones(w) 
        path= path* int(Window_LEN/2)
        last_p  = start_p
        path_cost = 0
        for i in range(w):
            
            #detemin search region
            k0=last_p-1;
            if(k0<0):
              k0=0
            k1=last_p+2;
            if(k1>h):
              k1=h
            # find the min point
            min=1000.0
            record_last = last_p
            for j in range(k0,k1):
              #diffrence = (img[j,i] -img[record_last,i-1])
              
              # using the differential is niceer than using the point value
              #because the diff can be used to multiply the path length
              #diffrence =  np.median(img[j,i:i+3]) - img[record_last,i-1]
              diffrence =  np.mean(img[j,i:i+5]) - img[record_last,i-1] 
              # calculte the step path lenth to multiply the differential
              varianc_pos = np.absolute(j-record_last)
              if( diffrence*(varianc_pos*0.4+1)<min):
                  min  = diffrence*(varianc_pos*0.4+1)
                  last_p=j 
            path_cost  = path_cost +  img[int(last_p),i]
            path[i]= last_p 
            
        return path,path_cost/w

    def find_the_starting(img):
        starting_piont=int(Window_LEN/2)
        new = img[:,0:Window_LEN]
        line=new.sum(axis=1)
        starting_piont =  np.argmin(line)
        return starting_piont

    def calculate_ave_mid(img):
        starting_piont=int(Window_LEN/2)
        h, w= img.shape

        new = img[:,0:2:w]
        line=new.sum(axis=1)
        mid_point =  np.argmin(line)

        return mid_point

    #apply deep learning to find the path
    def search_a_path_GPU(img): # input should be torch tensor
        #img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
        H,W= img.shape  #get size of image
        piece_num = 10 # the number of the squers 
        piece_W = int(W/piece_num)
        input_batch = np.zeros((piece_num,3,Resample_size,Resample_size)) # a batch with piece num

        for  slice_point in range (piece_num):
            img_piece = img[:,slice_point*piece_W:(slice_point+1)*piece_W]
            img_piece = cv2.resize(img_piece, (Resample_size,Resample_size), interpolation=cv2.INTER_AREA)
            input_batch[slice_point,0,:,:] = transform(img_piece)[0]
            input_batch[slice_point,1,:,:] = transform(img_piece)[0]
            input_batch[slice_point,2,:,:] = transform(img_piece)[0]
        input = torch.from_numpy(np.float32(input_batch)) 
        input = input.to(device) 

        
        #img2 = cv2.resize(img, (Resample_size,Resample_size), interpolation=cv2.INTER_AREA)
 
        #input3d =  np.zeros((3,Resample_size,Resample_size))
        #input3d[0,:,:]= transform(img2)[0]
        #input3d[1,:,:]= transform(img2)[0]
        #input3d[2,:,:]= transform(img2)[0]
        #input = torch.from_numpy(np.float32(input3d)) 
        #input = input.to(device) 

        inputv = Variable(input)

        #inputv = Variable(input.unsqueeze(0))
        output = netD(inputv)
        path_upsam = np.zeros(W)
        output = output.cpu().detach().numpy()
        for connect_point in range (piece_num):
            path_upsam[connect_point*piece_W:(connect_point+1)*piece_W] = signal.resample(
                output[connect_point,:], piece_W)
        path_upsam = path_upsam *Window_LEN
        #long_out  = np.append(np.flip(output),output)
        #long_out  = np.append(long_out,np.flip(output))
        #long_out = gaussian_filter1d (long_out ,1)
        #long_path_upsam  = signal.resample(long_out, 3*W)*Window_LEN
        #path_upsam = long_path_upsam[W:2*W]
        return path_upsam, 0


     
#test of this algorithm
#               
#frame = cv2.imread(operatedir)
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
##filtered_gray =    filter_img=cv2.GaussianBlur(gray,(10,10),3) 
#filtered_gray  = myfilter.gauss_filter_s (gray)
##filtered_gray =    filter_img=cv2.blur(gray,(7,7)) 
 
#cv2.imshow('initial',frame)
#cv2.imshow('gray',gray)
#cv2.imshow('filtered_gray',filtered_gray)
 
#path1,path_cost1=PATH.search_a_path(gray,40)
#path2,path_cost2 = PATH.search_a_path(filtered_gray,37)

#show1 =  gray
#show2 =  filtered_gray

#for i in  range (len(path1)):
#    show1[int(path1[i]),int(i)]=254
#show1 =  gray

#for i in range ( len(path2)):
#    show2[int(path2[i]),i]=254
 
#cv2.imshow('step_process',show1) 
#cv2.imshow('step_process_filter',show2) 

# Press Q on keyboard to  exit
           
     
    

    

    






    

    




