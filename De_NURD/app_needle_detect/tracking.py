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



class TRACKING(object):
    def __init__(self):
      
        self.start  = None
        self.end = None
        self.template  = None
        self.p_corr_m = None
    def trancking(self,p_start,p_end,start,end,result_img):
        backtorgb = cv2.cvtColor(result_img.astype(np.uint8),cv2.COLOR_GRAY2RGB)
        # whether it is the first of detection 
        if start is None and p_start is None :
            return None,None

        if (p_start is None) or (p_end is None):
            new_st = start
            new_ed = end 
        else:
            e_start , e_end = TRACKING.cross_corrlate( p_start,p_end, self.template, result_img)
           

            if start is not None :
                e_start = np.asarray(e_start)
                start = np.asarray(start)
                p_start = np.asarray(p_start)
                e_end = np.asarray(e_end)
                end = np.asarray(end)
                p_end = np.asarray(p_end)
                d_1 = math.sqrt((e_start[0] - start[0])**2 +  (e_start[1] - start[1])**2)
                d_2 = math.sqrt((e_end[0] - end[0])**2 +  (e_end[1] - end[1])**2)
                if (d_1 + d_2)/2 <50:
                    new_st =  (start + e_start )/2.0
                    new_ed =  (end + e_end )/2.0 
                 

                
                else:
                    new_st =  (e_start +p_start)/2.0
                    new_ed =  (e_end + p_end)/2.0 
            else:
                e_start = np.asarray(e_start)
           
                p_start = np.asarray(p_start)
                e_end = np.asarray(e_end)
            
                p_end = np.asarray(p_end)
                new_st =  (e_start +p_start)/2.0
                new_ed =  (e_end + p_end)/2.0 
            new_st = tuple( new_st.astype(int))
            new_ed = tuple( new_ed .astype(int))


        #conver line to a box 
        left= int(min(new_st[0],new_ed[0]))
        right= int(max(new_st[0],new_ed[0]))
        top= int( min(new_st[1],new_ed[1]))
        bottom= int(max(new_st[1],new_ed[1]))
        # update the template
        self.template  = result_img[top:bottom, left:right] 




        return new_st , new_ed
    def cross_corrlate( p_start,p_end,template, result_img):
        # define area 
        H,W  = result_img.shape

        left= min(p_start[0],p_end[0])
        right= max(p_start[0],p_end[0])
        top= min(p_start[1],p_end[1])
        bottom= max(p_start[1],p_end[1])

        left = np.clip(left-20,0,W)
        right = np.clip(right+20,0,W)
        top = np.clip(top-20,0,H)
        bottom = np.clip(bottom+20,0,H)
        template2 = template -template.mean()
        S_area  =   result_img[top:bottom, left:right] 
        #S_area2 =  S_area + np.random.randn(*S_area.shape) * 50 
        S_area2 =  S_area  -  S_area.mean ()

        cross_corr = signal.correlate2d(S_area2, template2,boundary='symm', mode='same') 
        cross_corr = cross_corr / np.max(cross_corr)
        cross_corr = np.clip(cross_corr *200,1,255)
        #full_corr_m  =

        y, x = np.unravel_index(np.argmax(cross_corr), cross_corr.shape) 
        y = y + top
        x = x + left

        shiftx = x - (p_start[0] + p_end[0])/2
        shifty = y - (p_start[1] + p_end[1])/2

     
        
       
        e_start= ( int(p_start[0] +  shiftx) , int(p_start[1] +shifty))
       
        e_end   = (int(p_end[0] +shiftx) , int(p_end[1] +  shifty))
    





        result_img = cv2.circle(result_img, (int(x),int(y)), radius=5, color= (255, 255, 255), thickness=-1)
        result_img = cv2.arrowedLine(result_img, e_start ,e_end, 
                                     (255, 255, 255)  , thickness=2,tipLength = 0.05)  
        #cv2.imshow('S_area',S_area.astype(np.uint8) ) 
        cv2.imshow('template',template.astype(np.uint8) ) 
        #cv2.imshow('cross_corr',cross_corr.astype(np.uint8) ) 
        cv2.imshow('result_img',result_img.astype(np.uint8) ) 


         

        return e_start , e_end