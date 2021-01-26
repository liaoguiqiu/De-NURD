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
        Line_detect.contour(edges,"edge contour")
        Line_detect.Hough (edges)
        cv2.imshow('edge',edges.astype(np.uint8) ) 
    def contour (thresh1,display = "conotur1"):
        kernel = np.ones((3,3),np.uint8)
        thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        thresh1 = thresh1.astype(np.uint8)
        _,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        backtorgb = cv2.cvtColor(thresh1,cv2.COLOR_GRAY2RGB)
        cv2.drawContours(backtorgb, contours, -1, (0,255,0), 3)
        backtorgb = Line_detect.select_contour(contours,  backtorgb)
        cv2.imshow(display,backtorgb.astype(np.uint8) ) 

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

    def select_contour (contours , backtorgb):

        number  = len (contours)

        for i in range( number): 
            this_contour  = contours[len(contours)-1]


        if len(contours) >0 :
            cnt = contours[len(contours)-1]
            rect = cv2.minAreaRect(cnt)
            area = cv2.contourArea(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            backtorgb = cv2.drawContours(backtorgb,[box],0,(0,0,255),2)
        return  backtorgb 
         
