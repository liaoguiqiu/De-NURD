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

        #ret2,thresh1 = cv2.threshold(result_img,200,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #thresh1 = cv2.adaptiveThreshold(result_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        #    cv2.THRESH_BINARY,11,2)
        ret,thresh1 = cv2.threshold(result_img,60,255,cv2.THRESH_BINARY)
        
        #opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
        #Line_detect.Hough (thresh1 ,display = " threshoud hough")

        #Line_detect.contour(thresh1)

        edges = cv2.Canny(result_img.astype(np.uint8),130,130,apertureSize = 3)
        lines  = Line_detect.Hough (edges)
        final_box,MP = Line_detect.contour(edges,lines,"edge contour")


        cv2.imshow('edge',edges.astype(np.uint8) ) 
        if final_box is not None:

            #backtorgb = cv2.drawContours(backtorgb,[final_box],0,(0,0,255),2)
            #cv2.line(backtorgb,(final_box[0][0],final_box[0][1]),(final_box[2][0],final_box[2][1]),(0,255,0),2)
            cv2.line(backtorgb,(MP[0][0],MP[0][1]),(MP[1][0],MP[1][1]),(0,255,0),2)

        cv2.imshow('Detection needle',backtorgb.astype(np.uint8) ) 
        return backtorgb


    def contour (thresh1,lines,display = "conotur1"):
        kernel = np.ones((3,3),np.uint8)
        #thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        thresh1 = thresh1.astype(np.uint8)
        _,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        backtorgb = cv2.cvtColor(thresh1,cv2.COLOR_GRAY2RGB)
        backtorgb = backtorgb*0
        cv2.drawContours(backtorgb, contours, -1, (255,255,255), 1)
        if contours is not None:
            backtorgb,final_box,final_MP = Line_detect.select_contour(contours,lines,  backtorgb)
        cv2.imshow(display,backtorgb.astype(np.uint8) ) 
        return final_box ,final_MP

    def Hough (edges,display = "hough1"):
        minLineLength = 3
        maxLineGap =3
        kernel = np.ones((2,2),np.uint8)
        #edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        #edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        #edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        #edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        #edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)


        lines = cv2.HoughLinesP(edges,1,np.pi/360,1,minLineLength,maxLineGap)
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
        satisfy = 0 
        final_MP =  np.zeros((2,2))
        for i in range( number): 
            cnt  = contours[i] # this contpo
            this_clen  = len(cnt)
            if (this_clen>30):
                rect = cv2.minAreaRect(cnt)
                width = int(rect[1][0])
                height = int(rect[1][1])
                area = cv2.contourArea(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                src_pts = box.astype("float32")

    # directly warp the rotated rectangle to get the straightened rectangle
    
    # coordinate of the points in box points after the rectangle has been
    # straightened
                dst_pts = np.array([[0, height-1],
                                    [0, 0],
                                    [width-1, 0],
                                    [width-1, height-1]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)

                L1 = math.sqrt((box[0][0] - box[1][0])**2 +  (box[0][1] - box[1][1])**2)
                L2 = math.sqrt((box[2][0] - box[1][0])**2 +  (box[2][1] - box[1][1])**2)

               

                # check it is long rectangual 
                 
                Long_rec = False 
                L_ratio = 1
                MP =  np.zeros((2,2))
                if L1 < L2:
                    # MP : merge the box into just one line: 
                    MP[0][0] = (box[0][0]+box[1][0])/2
                    MP[0][1] = (box[0][1]+box[1][1])/2
                    MP[1][0] = (box[2][0]+box[3][0])/2
                    MP[1][1] = (box[2][1]+box[3][1])/2

                    if L1/L2 < 0.25 and L2>40:
                        L_ratio   = L1/L2
                        Long_rec = True
                        
                else:
                    MP[0][0] = (box[0][0]+box[3][0])/2
                    MP[0][1] = (box[0][1]+box[3][1])/2
                    MP[1][0] = (box[2][0]+box[1][0])/2
                    MP[1][1] = (box[2][1]+box[1][1])/2
                    if L2/L1 < 0.25  and L1>40:
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

                #if ratio_err<0.3 and Long_rec ==True and Straight == True:
                if   Long_rec ==True and Straight == True:
                    satisfy += 1 
                    #err_sum = 0.5*L_ratio + 0.5 * Straight_ratio_error #+ 0.3 * ratio_err

                    backtorgb2  = cv2.cvtColor(backtorgb, cv2.COLOR_BGR2GRAY)




                    Mask  = backtorgb2 * 0
                    Mask2 = backtorgb2 * 0 +1
                    H,W = Mask2.shape

                    center_coordinates = (int(H/2), int(W/2))
# Radius of circle
                    radius = 50
  
                    # Blue color in BGR
                    #color = (255, 0, 0)
                    color = (0)

  
                    # Line thickness of 2 px
                    thickness = -1
                    
                    # Using cv2.circle() method
                    # Draw a circle with blue line borders of thickness of 2 px
                    Mask2 = cv2.circle(Mask2, center_coordinates, radius, color, thickness)
                    #cv2.imshow("warp" + str (satisfy),Mask2.astype(np.uint8) ) 


                    cv2.fillPoly(Mask, [box], (1 ) )

                    backtorgb2  = (backtorgb2>100)*1
                    backtorgb2 = backtorgb2 * Mask2
                    Select  = Mask * backtorgb2
                    backtorgb2 = backtorgb2 *255
                    backtorgb = cv2.cvtColor(backtorgb2.astype(np.uint8),cv2.COLOR_GRAY2RGB)

                    #warped = cv2.warpPerspective(backtorgb.astype(np.uint8), M, (width, height))
                    #warped  =   cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    EdgeLen  = np.sum (Select )
                    Select = Select *255
                    #cv2.imshow("warp" + str (satisfy),backtorgb.astype(np.uint8) * 0  )# back gorund first 
                    #cv2.imshow("warp" + str (satisfy),Select.astype(np.uint8) ) 
                    backtorgb = cv2.drawContours(backtorgb,[box],0,(0,0,255),1)
                    err_sum = np.abs(EdgeLen/2/(height + width) - 1)


                    if err_sum <Min_err and err_sum <0.2 :
                        final_box = box
                        Min_err = err_sum
                        final_MP =  MP.astype(np.int) 


        #if len(contours) >0 :
        #    cnt = contours[len(contours)-1]
            
        return  backtorgb, final_box,final_MP
         
