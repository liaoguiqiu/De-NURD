import cv2
import math
import numpy as np
import os
import torch
import scipy.signal as signal

#from numba import vectorize
#from numba import jit
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from scipy.ndimage import gaussian_filter1d 


def medfilt(x, k):
        """Apply a length-k median filter to a 1D array x.
        Boundaries are extended by repeating endpoints.
        """
        assert k % 2 == 1, "Median filter length must be odd."
        assert x.ndim == 1, "Input must be one-dimensional."
        k2 = (k - 1) // 2
        y = np.zeros ((len (x), k), dtype=x.dtype)
        y[:,k2] = x
        for i in range (k2):
            j = k2 - i
            y[j:,i] = x[:-j]
            y[:j,i] = x[0]
            y[:-j,-(i+1)] = x[j:]
            y[-j:,-(i+1)] = x[-1]
        return np.median (y, axis=1)

class myfilter:
    def median_filter_s(img):
        new = img 
        h, w= new.shape
      
        for i in range(h):
            new[i,:]= signal.medfilt(new[i,:],3)  
         
        return new
    def gauss_filter_s(img):
        new = img 
        h, w= new.shape
      
        for i in range(h):
            new[i,:]= gaussian_filter1d(new[i,:],3)  
         
        return new
    def steam_filter(steam):
       len,h,w = steam.shape
       new=np.zeros((h,w))
       #for i in range(h):
       #    for j in range(w):
       #        new[i,j] = np.median(steam[:,i,j] )  
       #new=steam[0,:,:] + steam[1,:,:]
       #new=new/2.0
       #torch.set_default_tensor_type('torch.cuda.FloatTensor')
       ten= torch.from_numpy(steam )
       ten=ten.to(device)
      #sort the tensor by frame sequence
       sort_ten,_=ten.sort(dim=0)
       mid= sort_ten[int(len/2),:,:]
       #a = steam[0,:,:]
       #b = steam[1,:,:]
       #c = steam[2,:,:]
       #new = middle(a,b,c)
       new =  torch.Tensor.cpu(mid).detach().numpy()
       return new
    def steam_filter_avg(steam):
       len,h,w = steam.shape
       new=np.zeros((h,w))
       #for i in range(h):
       #    for j in range(w):
       #        new[i,j] = np.median(steam[:,i,j] )  
       #new=steam[0,:,:] + steam[1,:,:]
       #new=new/2.0
       #torch.set_default_tensor_type('torch.cuda.FloatTensor')
       ten= torch.from_numpy(steam)
       image= torch.sum(ten,dim=0) / len
       new =  torch.Tensor.cpu(image).detach().numpy()
       #a = steam[0,:,:]
       #b = steam[1,:,:]
       #c = steam[2,:,:]
       #new = middle(a,b,c)
       return new


    

    



