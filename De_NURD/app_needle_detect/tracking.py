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
      
        self.remember_start  = None
        self.remember_end = None
        self.template  = None
        self.p_corr_m = None
        self.forgeting =0
    def trancking(self,p_start,p_end,start,end,result_img):
        backtorgb = cv2.cvtColor(result_img.astype(np.uint8),cv2.COLOR_GRAY2RGB)
        new_st = None
        new_ed = None

        # whether it is the first of detection 
        H,W   = result_img .shape
        if start is None and p_start is None :
            return None,None

        if self.remember_start is None :
            self.remember_start = start
            self.remember_end = end
        D=0
        if start is not None:
            D = math.sqrt((self.remember_start [0] - start[0])**2 +  (self.remember_start[1] - start[1])**2)
        
        if D <100:
            if (p_start is None) or (p_end is None):
                new_st = start
                new_ed = end 
            

                # update the template

                        #conver line to a box 
                left= int(min(new_st[0],new_ed[0]))
                right= int(max(new_st[0],new_ed[0]))
                top= int( min(new_st[1],new_ed[1]))
                bottom= int(max(new_st[1],new_ed[1]))
                left = np.clip(left,0,W)
                right = np.clip(right,0,W)
                top = np.clip(top,0,H)
                bottom = np.clip(bottom,0,H)
                self.template  = result_img[top:bottom, left:right] 
            else:
                e_start , e_end = self.cross_corrlate( p_start,p_end, self.template, result_img)
           

                if start is not None :
                    e_start = np.asarray(e_start)
                    start = np.asarray(start)
                    p_start = np.asarray(p_start)
                    e_end = np.asarray(e_end)
                    end = np.asarray(end)
                    p_end = np.asarray(p_end)
                    d_1 = math.sqrt((e_start[0] - start[0])**2 +  (e_start[1] - start[1])**2)
                    d_2 = math.sqrt((e_end[0] - end[0])**2 +  (e_end[1] - end[1])**2)
                    if (d_1 + d_2)/2 <100:
                        new_st =  (0.8 *start + 0.2 * e_start ) 
                        new_ed =  (0.8 *end + 0.2 * e_end ) 
                    
                        # update the template

                        #conver line to a box 
                        left= int(min(new_st[0],new_ed[0]))
                        right= int(max(new_st[0],new_ed[0]))
                        top= int( min(new_st[1],new_ed[1]))
                        bottom= int(max(new_st[1],new_ed[1]))
                        left = np.clip(left,0,W)
                        right = np.clip(right,0,W)
                        top = np.clip(top,0,H)
                        bottom = np.clip(bottom,0,H)
                        self.template  = result_img[top:bottom, left:right] 

                
                    else:
                        new_st =  (e_start +p_start)/2.0
                        new_ed =  (e_end + p_end)/2.0 
                else:
                    self.forgeting += 1

                    e_start = np.asarray(e_start)
                    e_end = np.asarray(e_end)
           
                    p_start = np.asarray(p_start)
                    p_end = np.asarray(p_end)

                    new_st =  (e_start +p_start)/2.0
                    new_ed =  (e_end + p_end)/2.0 
                    if self.forgeting >3:
                        self.forgeting =0
                        new_st =  None
                        new_ed =  None


            #new_st  = start
            #new_ed  = end

                if new_st is not None:
                    new_st = tuple( new_st.astype(int))
                    new_ed = tuple( new_ed .astype(int))


        
        # update the template
        if (start is not None) and (end is not None):
            pass
           




        return new_st , new_ed
    def cross_corrlate(self, p_start,p_end,template, result_img):
        # define area 
        H,W  = result_img.shape

        left= min(p_start[0],p_end[0])
        right= max(p_start[0],p_end[0])
        top= min(p_start[1],p_end[1])
        bottom= max(p_start[1],p_end[1])

        left = np.clip(left-5,0,W)
        right = np.clip(right+5,0,W)
        top = np.clip(top-5,0,H)
        bottom = np.clip(bottom+5,0,H)
        template2 = template -template.mean()
        S_area  =   result_img[top:bottom, left:right] 
        #S_area2 =  S_area + np.random.randn(*S_area.shape) * 50 
        S_area2 =  S_area  -  S_area.mean ()

        cross_corr = signal.correlate2d(S_area2, template2,boundary='symm', mode='same') 
        cross_corr = cross_corr  / np.max(cross_corr)
        #cross_corr = np.clip(cross_corr *200,1,255)
        full_corr_m  = result_img *0
        full_corr_m = full_corr_m.astype(float)
        full_corr_m [top:bottom, left:right]  = cross_corr
        full_corr_m = np.clip(full_corr_m ,0,255)
        if self.p_corr_m is None:

            fuse = full_corr_m
        else :
            fuse = full_corr_m + self.p_corr_m
            fuse = fuse  / np.max(fuse)
        
        self.p_corr_m  = fuse

        y, x = np.unravel_index(np.argmax(fuse), fuse.shape) 
        #y = y + top
        #x = x + left

        shiftx = x - (p_start[0] + p_end[0])/2
        shifty = y - (p_start[1] + p_end[1])/2

     
        
       
        e_start= ( int(p_start[0] +  shiftx) , int(p_start[1] +shifty))
       
        e_end   = (int(p_end[0] +shiftx) , int(p_end[1] +  shifty))
    





        result_img = cv2.circle(result_img, (int(x),int(y)), radius=5, color= (255, 255, 255), thickness=-1)
        result_img = cv2.arrowedLine(result_img, e_start ,e_end, 
                                     (255, 255, 255)  , thickness=2,tipLength = 0.05)  
        #cv2.imshow('S_area',S_area.astype(np.uint8) ) 
        cv2.imshow('template',template.astype(np.uint8) ) 
        cv2.imshow('cross_corr',(fuse*200).astype(np.uint8) ) 
        cv2.imshow('result_img',result_img.astype(np.uint8) ) 


         

        return e_start , e_end