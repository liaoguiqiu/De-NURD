#log of modification cre
# this  is used  to read json files and trasfer into a pkl file
import json as JSON
import cv2
import math
import numpy as np
import os
import random 
from zipfile import ZipFile
import scipy.signal as signal
import pandas as pd
from pathlib import Path

def mirror_similarity(tp):
    # this metric treate the middle line of the two vectors as the symmetric axis/ mirror
    def getFootPoint(point, line_p1, line_p2):
 
     # need to get the foot note on that line to mirror the points 
     #https://www.bbsmax.com/A/mo5kgb2L5w/
         x0 = point[0]
         y0 = point[1]

         x1 = line_p1[0]
         y1 = line_p1[1]

         x2 = line_p2[0]
         y2 = line_p2[1]
 
         k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1)) / \
             ((x2 - x1) ** 2 + (y2 - y1) ** 2 )*1.0
 
         xn = k * (x2 - x1) + x1
         yn = k * (y2 - y1) + y1
         return (xn, yn)
    # one pair 
    def sim_one_pair (p0,p1,p2,p3):
        # middle line of this pair 
        m0 = (p0+p2)/2.0
        m1 = (p1+p3)/2.0
        # mirror the points according to the lien 
        fp0 =  getFootPoint(p0,m0,m1)
        fp1 =  getFootPoint(p1,m0,m1)
        # symetric mirror the points
        sp0 = fp0 - p0 + fp0
        sp1 = fp1 - p1 + fp1

        dis = np.linalg.norm(sp0  - p2) + np.linalg.norm(sp1  - p3)
        L = (np.linalg.norm(p0  - p1) + np.linalg.norm(p2  - p3))/2.0
        s = L/(L + dis)
        return s
    s1 = sim_one_pair(tp[0],tp[1],tp[3],tp[2])
    s2 = sim_one_pair(tp[0],tp[3],tp[1],tp[2])

    return (s1+s2)/2.0

def Similarity(tp): # calculate the Eular similarity of the 4 points 
    # this metric is calculated with directly the Eular distance of of the array
    vetor1 = tp[0]-tp[1] 
    vetor2 = tp[3]-tp[2] # the same direction
    # see the page of 
    # https://blog.csdn.net/dcrmg/article/details/52416832
    err1= np.linalg.norm(vetor1 - vetor2) # the error of two vector c = a - b 

    L1 = (np.linalg.norm(vetor1)+ np.linalg.norm(vetor2))/2 # the average distance of the two vetors\
    # rotate to calculate another group of vector
    vetor3 = tp[1]-tp[2] 
    vetor4 = tp[0]-tp[3] # the same direction
    err2= np.linalg.norm(vetor3 - vetor4) # the error of two vector c = a - b 
    L2 = (np.linalg.norm(vetor3)+ np.linalg.norm(vetor4))/2 # the average distance of the two vetors
    S  = (L1 + L2)/(L1+L2+err1+err2)   # Eular similarity
    return S

def err_90(tp): # calculate the Eular similarity of the 4 points 
    def angle_err(vetor1,vetor2):
        unit_vector_1 = vetor1 / np.linalg.norm(vetor1)
        unit_vector_2 = vetor2 / np.linalg.norm(vetor2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = abs(np.arccos(dot_product) *180/3.14159)
        err = abs(90-angle )
        return err
    vetor1 = tp[0]-tp[1] 
    vetor2 = tp[1]-tp[2] # the same direction
    vetor3 = tp[2]-tp[3] 
    vetor4 = tp[3]-tp[0] 
    err1 = angle_err(vetor1,vetor2)
    err2 = angle_err(vetor2,vetor3)
    err3 = angle_err(vetor3,vetor4)
    err4 = angle_err(vetor1,vetor4)
    return sum([err1,err2,err3,err4]) 

def overall_angle(tp0,tp1): # the overall anggle between inital and now
    def angle_cal(vetor1,vetor2):
        unit_vector_1 = vetor1 / np.linalg.norm(vetor1)
        unit_vector_2 = vetor2 / np.linalg.norm(vetor2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = abs(np.arccos(dot_product) *180/3.14159)
        return angle
    vetor01  = tp0[0]-tp0[2] 
    vetor02  = tp0[1]-tp0[3] # the same direction
    vetor11  = tp1[0]-tp1[2] 
    vetor12  = tp1[1]-tp1[3] # the same direction
    angle =(angle_cal(vetor01, vetor11)+ angle_cal(vetor02, vetor12))/2
    return angle
     
    
class  Geometry_Analy(object):
    def __init__(self ):
        #self.image_dir   = "../../OCT/beam_scanning/Data set/pic/NORMAL-BACKSIDE-center/"
        #self.roi_dir =  "../../OCT/beam_scanning/Data set/seg label/NORMAL-BACKSIDE-center/"
        #self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL/"
        #self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL-BACKSIDE-center/"
        #self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL-BACKSIDE/"
        
        #self.json_dir ="E:/database/NURD/20th October/correct/wire1/label/"
        #self.json_dir = "E:/database/NURD/8th 10 2021 colection for MedIA/correct/simplep1/label/"
        self.json_dir = "E:/database/NURD/8th 10 2021 colection for MedIA/correct/simplep1/label/"
        #base_dir =  os.path.basename(os.path.normpath(self.json_dir))
        path = Path(self.json_dir)
        self.save_dir =  str(path.parent.absolute()) + "/excel_result/"
        try:
            os.stat(self.save_dir)
        except:
            os.mkdir(self.save_dir)
        folder_list = os.listdir(self.json_dir)
        self.similarity_buff =np.zeros((len(folder_list),2)) 
        self.mirror_similarity_buff =np.zeros((len(folder_list),2)) 

        self.err90_buff =np.zeros((len(folder_list),2)) 
        self.all_angle_buff =np.zeros((len(folder_list),2)) 


        self.originalarray =  [] # no predefines # predefine there are 4 contours
        self.correctedarray =  [] # predefine there are 4 contours
        self.array_original = []
        self.array_corrected  = []
        self.recordedflag = 0
    def all_angle_err(self,tp1,tp2):
        if self.recordedflag == 0 :
           self.array_original =  tp1
           self.array_corrected =  tp2
           self.recordedflag =1
        else:
            err1  = overall_angle( self.array_original, tp1)
            err2 =  overall_angle( self.array_corrected, tp2)
            return err1,err2
    def check_one_folder (self):
        folder_list = os.listdir(self.json_dir)
        data_i = 0
        for i in folder_list:
   
            a, b = os.path.splitext(i)
            # if it is a json it will have corresponding image 
            if  b == ".json":
                #img_path = self.image_dir + a + ".jpg"
                #img1 = cv2.imread(img_path)
                #if img1 is None:
                #    print ("no_img for this zip")
                #else:
                    json_dir = self.json_dir + a + b
                    with open(json_dir) as f_dir:
                        data = JSON.load(f_dir)
                    shape  = data["shapes"]
                    num_line  = len(shape)
                    len_list=  num_line
                    #with ZipFile(json_dir, 'r') as zipObj:
                    #       # Get list of files names in zip
                    #       #listOfiles = zipObj.namelist()
                    #       # this line of code is importanct sice the the formmer one will change the sequence 
                    #       listOfiles = zipObj.infolist()
                    #       len_list = len(listOfiles)

                           
                    #rois = read_roi_zip(roi_dir) # get all the coordinates
                    for iter in range(len_list):

                        if shape[iter]["label"] == 'original':
                            tp1 = np.array(shape[iter]["points"])

                            S_ori = Similarity(tp1)
                            MS_ori = mirror_similarity(tp1)
                            angle_err_or = err_90(tp1)
 

                        else:
                            tp2 = np.array(shape[iter]["points"])

                            S_corr = Similarity(tp2)
                            MS_corr = mirror_similarity(tp2)

                            angle_err_co = err_90(tp2)
                            pass
                    all_an_err = self.all_angle_err(tp1,tp2)
                    self.similarity_buff[data_i] = [S_ori,S_corr]
                    self. err90_buff [data_i] =  [angle_err_or,angle_err_co]
                    self. all_angle_buff [data_i] = all_an_err
                    self. mirror_similarity_buff[data_i]=  [MS_ori, MS_corr]

                    DF1 = pd.DataFrame(self.similarity_buff)
                    DF1.to_csv(self.save_dir+"similarity_buff.csv")
                    DF2 = pd.DataFrame(self.err90_buff)
                    DF2.to_csv(self.save_dir+"err90_buff.csv")
                    DF3 = pd.DataFrame(self.all_angle_buff)
                    DF3.to_csv(self.save_dir+"all_angle_buff.csv")
                    DF4 = pd.DataFrame(self.mirror_similarity_buff)
                    DF4.to_csv(self.save_dir+"mirror_similarity_buff.csv")
                    print(str(data_i))
# save the dataframe as a csv file
                    

                    data_i+=1
            pass

if __name__ == '__main__':
   

    calculator  = Geometry_Analy()
    calculator.check_one_folder() # convert the Json file into pkl files 


