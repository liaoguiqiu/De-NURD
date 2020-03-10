import cv2
import math
import numpy as np
import os
import torch
import scipy.signal as signal
from scipy.stats.stats import pearsonr   
from scipy.ndimage import gaussian_filter1d

#from numba import vectorize,float32
#from numba import cuda

#from numba import jit

##@vectorize([float32(float32, float32)])
##def correlation_GPU(a, b):
##    result  = pearsonr(a,b)
##    return  result
#@jit
#def correlation_GPU(a, b):
#    result  = pearsonr(a,b)
#    return  result

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Window_LEN = 71
Overall_shiftting_WinLen = 71
class COSTMtrix:

    def matrix_cal_corre(sequence):

       window_wid= Window_LEN
       window_cntr= int(Window_LEN/2)  # check
       len,h,w = sequence.shape
       present_img = sequence[len-1,:,:]
       previous_img = sequence[len-2,:,:]
       matrix = np.zeros ((window_wid, w))
       #main loop (for every scanning line)
       for i in range(window_cntr,w-window_cntr): # check the ending for index
           for j in range(window_wid): #sub_loop for shift distance

               #a=present_img[:,i]-previous_img[:,i-window_cntr+j]
               #a= np.sqrt(a*a)
               #ele=np.sum(a)/h
               a = pearsonr(present_img[:,i],previous_img[:,i-window_cntr+j])

               #a1 = present_img[:,i:i+3]
               #a2  = previous_img[:,i-window_cntr+j:i-window_cntr+j+3]
               #a1 = a1.flatten()
               #a2= a2.flatten()
               #a = pearsonr(a1,a2)

               #matrix[j,i] = ele*ele/3
               #matrix[j,i] = ele*ele/3

               matrix[j,i] = 251-a[0]*250
               
                
       return matrix
###################
# use the delayed one to realize full image correction, and 
#make the shifting window center variable
    def matrix_cal_corre_full_version_2(sequence,window_shift):

       window_wid= Window_LEN
       window_cntr= int(Window_LEN/2)  # check
       len,h,w = sequence.shape

       present_img = sequence[len-1,:,:]
       previous_img = sequence[len-2,:,:]
       pre_previous = sequence[len-3,:,:]
       #connect 3 scanning images together to make the correlation can be done out of the boundary
       add_3_img  = np.append(previous_img,previous_img,axis=1) # cascade
       add_3_img = np.append(add_3_img,previous_img,axis=1) # cascade
       matrix = np.zeros ((window_wid, w))
       #main loop (for every scanning line)
       for i in range(w): # check the ending for index
           for j in range(window_wid): #sub_loop for shift distance
               a = pearsonr(present_img[:,i],add_3_img[:,i-window_cntr+j+w + int(window_shift)])
               matrix[j,i] = 251-a[0]*250    
       return matrix,int(window_shift)
###################
# use the delayed one to realize full image correction
    def matrix_cal_corre_full_version_2GPU(sequence,window_shift):

       window_wid= Window_LEN
       window_cntr= int(Window_LEN/2)  # check
       len,h,w = sequence.shape

       present_img = sequence[len-1,:,:]
       #previous_img = sequence[len-2,:,:] #  use the corrected  near img
       previous_img = sequence[len-2,:,:] #  use the first Img

       pre_previous = sequence[len-3,:,:]
       #connect 3 scanning images together to make the correlation can be done out of the boundary
       add_3_img  = np.append(previous_img,previous_img,axis=1) # cascade
       add_3_img = np.append(add_3_img,previous_img,axis=1) # cascade
       matrix = np.zeros ((window_wid, w))
       a_stack= np.zeros((window_wid,w,h))
       b_stack= np.zeros((window_wid,w,h))

       #main loop (for every scanning line)
       for i in range(w): # check the ending for index
           for j in range(window_wid): #sub_loop for shift distance
               a_stack[j,i,:] =present_img[:,i]
               b_stack[j,i,:] =add_3_img[:,i-window_cntr+j+w + int(window_shift)]
               #do stack beforcaculation
               # arow staxk
               #if (j==0):
               #    a_row_stack = present_img[:,i]
               #else:
               #    a_row_stack = np.vstack( (a_row_stack,present_img[:,i]))
               ##b_row stack   
               #if (j==0):
               #    b_row_stack = add_3_img[:,i-window_cntr+j+w + int(window_shift)]
               #else:
               #    b_row_stack = np.vstack(( b_row_stack,add_3_img[:,i-window_cntr+j+w + int(window_shift)])) 
            #complete stack
           #if (i==0):
           #     a_stack = [a_row_stack]
           #else:
           #     a_stack = np.vstack( (a_stack,[a_row_stack]))
           # #b_row stack   
           #if (i==0):
           #     b_stack = [b_row_stack]
           #else:
           #     b_stack = np.vstack(( b_stack,[b_row_stack])) 
               #a = correlation_GPU(present_img[:,i],add_3_img[:,i-window_cntr+j+w + int(window_shift)])
               #matrix[j,i] = 251-a[0]*250  
       #matrix  = correlation_GPU (a_stack  , b_stack)
       a_stack  =  torch.from_numpy(a_stack)
       b_stack  =  torch.from_numpy(b_stack)
       #mulab = torch.mul(a_stack,b_stack)
       #mula2 = torch.mul(a_stack , a_stack)
       #mulb2 = torch.mul(b_stack , b_stack)
       #suma = torch.sum(a_stack,dim=2)
       #sumb = torch.sum(b_stack,dim=2)
       #sumab = torch.sum(mulab,dim=2)
       #suma2 = torch.sum(mula2,dim=2)
       #sumb2 = torch.sum(mulb2,dim=2)
       #correlation_Mat= (h*sumab - suma*sumb)/ torch.sqrt((h*suma2-suma*suma)*(h*sumb2-sumb*sumb))
       #correlation_Mat =  251 - correlation_Mat*250
       #mulab = torch.mul(a_stack,b_stack)
       #mula2 = torch.mul(a_stack , a_stack)
       #mulb2 = torch.mul(b_stack , b_stack)
       suma = torch.sum(a_stack,dim=2)
       sumb = torch.sum(b_stack,dim=2)
       sumab = torch.sum(a_stack*b_stack,dim=2)
       suma2 = torch.sum(a_stack*a_stack,dim=2)
       sumb2 = torch.sum(b_stack*b_stack,dim=2)
       correlation_Mat= (h*sumab - suma*sumb)/ torch.sqrt((h*suma2-suma*suma)*(h*sumb2-sumb*sumb))
       correlation_Mat =  251 - correlation_Mat*250


       # copy frome the GPU
       matrix=torch.Tensor.cpu(correlation_Mat).detach().numpy()
       return matrix,int(window_shift)
#########################
###################
# use the delayed one to realize full image correction line correlation 
    def matrix_cal_corre_full_version3_2GPU(present_img,previous_img,window_shift):

       window_wid= Window_LEN
       window_cntr= int(Window_LEN/2)  # check
       h,w = present_img.shape

       #present_img = sequence[len-1,:,:]
       ##previous_img = sequence[len-2,:,:] #  use the corrected  near img
       #previous_img = sequence[0,:,:] #  use the first Img
       #connect 3 scanning images together to make the correlation can be done out of the boundary
       add_3_img  = np.append(previous_img,previous_img,axis=1) # cascade
       add_3_img = np.append(add_3_img,previous_img,axis=1) # cascade
       matrix = np.zeros ((window_wid, w))
       a_stack= np.zeros((window_wid,w,h))
       b_stack= np.zeros((window_wid,w,h))

       #main loop (for every scanning line)
       for i in range(w): # check the ending for index
           for j in range(window_wid): #sub_loop for shift distance
               a_stack[j,i,:] =present_img[:,i]
               b_stack[j,i,:] =add_3_img[:,i-window_cntr+j+w + int(window_shift)]
               #do stack beforcaculation
               # arow staxk
               #if (j==0):
               #    a_row_stack = present_img[:,i]
               #else:
               #    a_row_stack = np.vstack( (a_row_stack,present_img[:,i]))
               ##b_row stack   
               #if (j==0):
               #    b_row_stack = add_3_img[:,i-window_cntr+j+w + int(window_shift)]
               #else:
               #    b_row_stack = np.vstack(( b_row_stack,add_3_img[:,i-window_cntr+j+w + int(window_shift)])) 
            #complete stack
           #if (i==0):
           #     a_stack = [a_row_stack]
           #else:
           #     a_stack = np.vstack( (a_stack,[a_row_stack]))
           # #b_row stack   
           #if (i==0):
           #     b_stack = [b_row_stack]
           #else:
           #     b_stack = np.vstack(( b_stack,[b_row_stack])) 
               #a = correlation_GPU(present_img[:,i],add_3_img[:,i-window_cntr+j+w + int(window_shift)])
               #matrix[j,i] = 251-a[0]*250  
       #matrix  = correlation_GPU (a_stack  , b_stack)
       a_stack  =  torch.from_numpy(a_stack)
       b_stack  =  torch.from_numpy(b_stack)
       #mulab = torch.mul(a_stack,b_stack)
       #mula2 = torch.mul(a_stack , a_stack)
       #mulb2 = torch.mul(b_stack , b_stack)
       #suma = torch.sum(a_stack,dim=2)
       #sumb = torch.sum(b_stack,dim=2)
       #sumab = torch.sum(mulab,dim=2)
       #suma2 = torch.sum(mula2,dim=2)
       #sumb2 = torch.sum(mulb2,dim=2)
       #correlation_Mat= (h*sumab - suma*sumb)/ torch.sqrt((h*suma2-suma*suma)*(h*sumb2-sumb*sumb))
       #correlation_Mat =  251 - correlation_Mat*250
       #mulab = torch.mul(a_stack,b_stack)
       #mula2 = torch.mul(a_stack , a_stack)
       #mulb2 = torch.mul(b_stack , b_stack)
       suma = torch.sum(a_stack,dim=2)
       sumb = torch.sum(b_stack,dim=2)
       sumab = torch.sum(a_stack*b_stack,dim=2)
       suma2 = torch.sum(a_stack*a_stack,dim=2)
       sumb2 = torch.sum(b_stack*b_stack,dim=2)
       correlation_Mat= (h*sumab - suma*sumb)/ torch.sqrt((h*suma2-suma*suma)*(h*sumb2-sumb*sumb))
       correlation_Mat =  251 - correlation_Mat*250


       # copy frome the GPU
       matrix=torch.Tensor.cpu(correlation_Mat).detach().numpy()
       return matrix,int(window_shift)
#########################
###################
# use the delayed one to realize full image correction line correlation 
    def matrix_line_distance_version3_2GPU(present_img,previous_img,window_shift):

       window_wid= Window_LEN
       window_cntr= int(Window_LEN/2)  # check
       h,w = present_img.shape

       #connect 3 scanning images together to make the correlation can be done out of the boundary
       add_3_img  = np.append(previous_img,previous_img,axis=1) # cascade
       add_3_img = np.append(add_3_img,previous_img,axis=1) # cascade
       matrix = np.zeros ((window_wid, w))
       a_stack= np.zeros((window_wid,w,h))
       b_stack= np.zeros((window_wid,w,h))

       #main loop (for every scanning line)
       for i in range(w): # check the ending for index
           for j in range(window_wid): #sub_loop for shift distance
               a_stack[j,i,:] =present_img[:,i]
               b_stack[j,i,:] =add_3_img[:,i-window_cntr+j+w + int(window_shift)]
               
       a_stack  =  torch.from_numpy(a_stack)
       b_stack  =  torch.from_numpy(b_stack)
       error = a_stack - b_stack
      
       sum = torch.sum(error*error/h,dim=2)
        
       distance_Mat=  torch.sqrt(sum)
        


       # copy frome the GPU
       #matrix=torch.Tensor.cpu(correlation_Mat).detach().numpy()
    
       matrix=torch.Tensor.cpu(distance_Mat).detach().numpy()
       return matrix,int(window_shift)
#########################
###################
# use the delayed one to realize full image correction
    def Img_fully_shifting_correlation(present_img,previous_img,window_shift):

       window_wid= Overall_shiftting_WinLen
       window_cntr= int(Overall_shiftting_WinLen/2)  # check
       h,w = present_img.shape

       #present_img = sequence[len-1,:,:]
       ##previous_img = sequence[len-2,:,:] #  use the corrected  near img
       #previous_img = sequence[0,:,:] #  use the first Img

       #pre_previous = sequence[len-3,:,:]
       #connect 3 scanning images together to make the correlation can be done out of the boundary
       add_3_img  = np.append(previous_img,previous_img,axis=1) # cascade
       add_3_img = np.append(add_3_img,previous_img,axis=1) # cascade
       matrix = np.zeros (window_wid )
       a_stack= np.zeros((window_wid,h,w))
       b_stack= np.zeros((window_wid,h,w))

       #main loop (for every scanning line)
       #for i in range(w): # check the ending for index
       for j in range(window_wid): #sub_loop for shift distance
            a_stack[j,:,:] =present_img[:,:]  # dupicate for many time for shifting correlation
            # shifting cropping for connected image
            crop_start  = -window_cntr+j+w + int(window_shift)
            crop_end  =  crop_start + w
            b_stack[j,:,:] =add_3_img[:,crop_start:crop_end]
       a_stack  =  torch.from_numpy(a_stack)
       b_stack  =  torch.from_numpy(b_stack)  
       suma = torch.sum(a_stack,dim=2)
       sumb = torch.sum(b_stack,dim=2)
       sumab = torch.sum(a_stack*b_stack,dim=2)
       suma2 = torch.sum(a_stack*a_stack,dim=2)
       sumb2 = torch.sum(b_stack*b_stack,dim=2)

       suma  = torch.sum (suma , dim =1)
       sumb = torch.sum(sumb,dim=1)
       sumab = torch.sum(sumab,dim=1)
       suma2 = torch.sum(suma2,dim=1)
       sumb2 = torch.sum(sumb2,dim=1)


       correlation_Mat= (h*w*sumab - suma*sumb)/ torch.sqrt((h*w*suma2-suma*suma)*(h*w*sumb2-sumb*sumb))
       correlation_Mat =  251 - correlation_Mat*250


       # copy frome the GPU
       matrix=torch.Tensor.cpu(correlation_Mat).detach().numpy()
 
 

 
       matrix = gaussian_filter1d(matrix,3) # smooth the path 
       mid_point =  np.argmin(matrix)
       return mid_point,int(window_shift)
#########################
###################
# use the delayed one to realize full image correction
    def Img_fully_shifting_distance(present_img,previous_img,window_shift):

       window_wid= Window_LEN
       window_cntr= int(Window_LEN/2)  # check
       h,w = present_img.shape

       #present_img = sequence[len-1,:,:]
       ##previous_img = sequence[len-2,:,:] #  use the corrected  near img
       #previous_img = sequence[0,:,:] #  use the first Img

       #pre_previous = sequence[len-3,:,:]
       #connect 3 scanning images together to make the correlation can be done out of the boundary
       add_3_img  = np.append(previous_img,previous_img,axis=1) # cascade
       add_3_img = np.append(add_3_img,previous_img,axis=1) # cascade
       matrix = np.zeros (window_wid )
       a_stack= np.zeros((window_wid,h,w))
       b_stack= np.zeros((window_wid,h,w))

       #main loop (for every scanning line)
       #for i in range(w): # check the ending for index
       for j in range(window_wid): #sub_loop for shift distance
            a_stack[j,:,:] =present_img[:,:]  # dupicate for many time for shifting correlation
            # shifting cropping for connected image
            crop_start  = -window_cntr+j+w + int(window_shift)
            crop_end  =  crop_start + w
            b_stack[j,:,:] =add_3_img[:,crop_start:crop_end]
       a_stack  =  torch.from_numpy(a_stack)
       b_stack  =  torch.from_numpy(b_stack)         
       error = a_stack - b_stack
       error = error*error
       error_sum1  = torch.sum(error ,dim=2) 
       error_sum2  = torch.sum(error_sum1,dim=1) 

       # copy frome the GPU
       error_sum2=torch.Tensor.cpu(error_sum2).detach().numpy()
       error_sum2 = gaussian_filter1d(error_sum2,7) # smooth the path 
       mid_point =  np.argmin(error_sum2)
       return mid_point,int(window_shift)
#########################
    def matrix_cal_Euler(sequence):

        window_wid= Window_LEN
        window_cntr= int(Window_LEN/2)  # check
        len,h,w = sequence.shape
        present_img = sequence[len-1,:,:]
        previous_img = sequence[len-2,:,:]
        matrix = np.zeros ((window_wid, w))
        #main loop (for every scanning line)
        for i in range(window_cntr,w-window_cntr): # check the ending for index
            for j in range(window_wid): #sub_loop for shift distance

                a=present_img[:,i]-previous_img[:,i-window_cntr+j]
                a= np.sqrt(a*a)
                ele=np.sum(a)/h                         
                matrix[j,i] = ele 
                
        matrix =  250*matrix / np.amax(matrix)                 
        return matrix
    #########################
    def matrix_cal_Euler_GPU(sequence,window_shift):

       window_wid= Window_LEN
       window_cntr= int(Window_LEN/2)  # check
       len,h,w = sequence.shape

       present_img = sequence[len-1,:,:]
       previous_img = sequence[len-2,:,:]
       pre_previous = sequence[len-3,:,:]
       #connect 3 scanning images together to make the correlation can be done out of the boundary
       add_3_img  = np.append(previous_img,previous_img,axis=1) # cascade
       add_3_img = np.append(add_3_img,previous_img,axis=1) # cascade
       matrix = np.zeros ((window_wid, w))
       a_stack= np.zeros((window_wid,w,h))
       b_stack= np.zeros((window_wid,w,h))

       #main loop (for every scanning line)
       for i in range(w): # check the ending for index
           for j in range(window_wid): #sub_loop for shift distance
               a_stack[j,i,:] =present_img[:,i]
               b_stack[j,i,:] =add_3_img[:,i-window_cntr+j+w + int(window_shift)]
               
       a_stack  =  torch.from_numpy(a_stack)
       b_stack  =  torch.from_numpy(b_stack)
       error = a_stack - b_stack
      
       sum = torch.sum(error*error/h,dim=2)
        
       correlation_Mat=  torch.sqrt(sum)
        


       # copy frome the GPU
       #matrix=torch.Tensor.cpu(correlation_Mat).detach().numpy()
       matrix=correlation_Mat

       return matrix,int(window_shift)
#########################

    

    






    

    



