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

class Line_detect:
    def detection(result_img):
        backtorgb = cv2.cvtColor(result_img.astype(np.uint8),cv2.COLOR_GRAY2RGB)
        result_img = cv2.GaussianBlur(result_img,(5,5),0)
        result_img =   result_img.astype(np.uint8)

        ret2,thresh1 = cv2.threshold(result_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #thresh1 = cv2.adaptiveThreshold(result_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        #    cv2.THRESH_BINARY,11,2)
        #ret,thresh1 = cv2.threshold(result_img,50,255,cv2.THRESH_BINARY)
        
        #opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
        #Line_detect.Hough (thresh1 ,display = " threshoud hough")

        #Line_detect.contour(thresh1)

        edges = cv2.Canny(result_img.astype(np.uint8),50,100,apertureSize = 3)
        lines  = Line_detect.Hough (edges)
        final_box = Line_detect.contour(edges,lines,"edge contour")


        cv2.imshow('edge',edges.astype(np.uint8) ) 
        if final_box is not None:
            backtorgb = cv2.drawContours(backtorgb,[final_box],0,(0,0,255),2)
        cv2.imshow('Detection needle',backtorgb.astype(np.uint8) ) 
        return backtorgb


    def contour (thresh1,lines,display = "conotur1"):
        kernel = np.ones((3,3),np.uint8)
        thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        thresh1 = thresh1.astype(np.uint8)
        _,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        backtorgb = cv2.cvtColor(thresh1,cv2.COLOR_GRAY2RGB)
        cv2.drawContours(backtorgb, contours, -1, (0,255,0), 3)
        if contours is not None:
            backtorgb,final_box = Line_detect.select_contour(contours,lines,  backtorgb)
        cv2.imshow(display,backtorgb.astype(np.uint8) ) 
        return final_box

    def Hough (edges,display = "hough1"):
        minLineLength = 30
        maxLineGap =20
        kernel = np.ones((2,2),np.uint8)
        #edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        #edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        #edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        #edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        #edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)


        lines = cv2.HoughLinesP(edges,1,np.pi/180,30,minLineLength,maxLineGap)
        result_img =  cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
        if lines is not None:
            number,_,_ = lines.shape
            for ii in range(number):

                for x1,y1,x2,y2 in lines[ii] :
                    cv2.line(result_img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.imshow(display,result_img.astype(np.uint8) ) 
        return lines
    def select_contour (contours ,lines, backtorgb):

        number  = len (contours)
        Min_Lratio  = 1
        Min_ratio_err  = 1 
        Min_Straight_ratio_error =1
        Min_err =10
        final_box = None
        for i in range( number): 
            cnt  = contours[i] # this contpo
            this_clen  = len(cnt)
            if (this_clen>30):
                rect = cv2.minAreaRect(cnt)
                area = cv2.contourArea(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                L1 = math.sqrt((box[0][0] - box[1][0])**2 +  (box[0][1] - box[1][1])**2)
                L2 = math.sqrt((box[2][0] - box[1][0])**2 +  (box[2][1] - box[1][1])**2)

               

                # check it is long rectangual 
                Long_rec = False 
                L_ratio = 1
                if L1 < L2:
                    if L1/L2 < 0.2 and L2>50:
                        L_ratio   = L1/L2
                        Long_rec = True
                        
                else:
                    if L2/L1 < 0.2  and L1>50:
                        L_ratio   = L2/L1
                        Long_rec = True
                # is straight or not :
                Straight = False

                Straight_ratio_error = 1
                if lines is not None:
                    left = np.min(box[:,0])
                    right = np.max(box[:,0])
                    top = np.min(box[:,1])
                    bottom = np.max(box[:,1])
                    line_len  = len(lines)
                    for j in  range(line_len) : # any line inside the rectangular
                        thisline = lines[j][0]
                        if (thisline[0]>left and thisline[0]<right and thisline[2]>left and thisline[2]<right):
                            if (thisline[1]>top and thisline[1]<bottom and thisline[3]>top and thisline[3]<bottom):
                                Straight_line_len = math.sqrt((thisline[0] - thisline[2])**2 +  (thisline[1] - thisline[3])**2)
                                Straight_ratio_error = np.abs (1 -Straight_line_len/(L1+L2))
                                Straight = True
                                break
                     


                ratio   = this_clen /2/(L1 + L2) # check the ratio nbetwee the contour lent and rectangualr perimeter
                ratio_err = np.abs(ratio - 1)

                if ratio_err<0.2 and Long_rec ==True and Straight == True:
                    err_sum = 0.3*L_ratio + 0.3 * Straight_ratio_error + 0.3 * ratio_err
                    

                    backtorgb = cv2.drawContours(backtorgb,[box],0,(0,0,255),2)
                    if err_sum <Min_err:
                        final_box = box
                        Min_err = err_sum


        #if len(contours) >0 :
        #    cnt = contours[len(contours)-1]
            
        return  backtorgb, final_box
         
