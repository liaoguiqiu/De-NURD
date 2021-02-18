# Functions on path post processing
import cv2
import math
import numpy as np
from median_filter_special import  myfilter

import os
import torch
import scipy.signal as signal
import pandas as pd

import random
   
class PATH_POST:
    def path_integral(P_hat,delta_P):
        #every line will be moved to a new postion
        l1 = len(P_hat)
        add_3   = np.append(P_hat ,P_hat,axis=0) # cascade
        add_3   = np.append(add_3,P_hat,axis=0) # cascade
        add_3_delta   = np.append(delta_P ,delta_P,axis=0) # cascade
        add_3_delta   = np.append(add_3_delta,delta_P,axis=0) # cascade

        new_P  = add_3*np.nan 



        for i in range ( len(add_3)):
            #limit the integral             
            new_position  = int(add_3_delta[i]+i)
            # deal with the boundary exceeding
            if(new_position<0):
                new_position = 0
            if (new_position>=3*l1):
                new_position= 3*l1-1
            #move this line to new position
            new_P[ i] = add_3_delta[i] + add_3[new_position] 
            #mask[:,int(new_position)] = 0
        #add_3   = np.append(path_inv[::-1],path_inv,axis=0) # cascade
        #add_3   = np.append(add_3,path_inv[::-1],axis=0) # cascade
        s = pd.Series(new_P)
        new_P = s.interpolate()
        path = new_P[l1:2*l1]
         

        return path
