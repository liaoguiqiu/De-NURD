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
from generator_contour import Save_Contour_pkl

 
class  Geometry_Analy(object):
    def __init__(self ):
        #self.image_dir   = "../../OCT/beam_scanning/Data set/pic/NORMAL-BACKSIDE-center/"
        #self.roi_dir =  "../../OCT/beam_scanning/Data set/seg label/NORMAL-BACKSIDE-center/"
        #self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL/"
        #self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL-BACKSIDE-center/"
        #self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL-BACKSIDE/"
        
        self.json_root = "D:/Deep learning/dataset/label data/"

       
         
        self.contours_x =  [] # no predefines # predefine there are 4 contours
        self.contours_y =  [] # predefine there are 4 contours

        self.saver = Save_Contour_pkl()
        self.display_flag = True
    def draw_coordinates_color(self,img1,vx,vy,color):
        
        if color ==0:
           painter  = [254,0,0]
        elif color ==1:
           painter  = [0,254,0]
        elif color ==2:
           painter  = [0,0,254]
        else :
           painter  = [0,0,0]
                    #path0  = signal.resample(path0, W)
        H,W,_ = img1.shape
        for j in range (len(vx)):
                #path0l[path0x[j]]
                dy = np.clip(vy[j],2,H-2)
                dx = np.clip(vx[j],2,W-2)

                img1[int(dy)+1,int(dx),:]=img1[int(dy),int(dx),:]=painter
                #img1[int(dy)+1,dx,:]=img1[int(dy)-1,dx,:]=img1[int(dy),dx,:]=painter


        return img1
    def check_one_folder (self):
        for i in os.listdir(self.json_dir):
    #for i in os.listdir("E:\\estimagine\\vs_project\\PythonApplication_data_au\\pic\\"):
        # separath  the name of json 
            a, b = os.path.splitext(i)
            # if it is a json it will have corresponding image 
            if b == ".json" :
                img_path = self.image_dir + a + ".jpg"
                img1 = cv2.imread(img_path)
                if img1 is None:
                    print ("no_img for this zip")
                else:
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
                    gray  =   cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # transfer into gray image
                    H,W   = gray.shape
                    max_buffer  = np.zeros(len_list)
                    contoursx=[None]*len_list
                    contoursy=[None]*len_list

                    for iter in range(len_list):
                        # get the name of one contour 
                        #  iter
                        coordinates  = shape[iter]["points"]
                        coordinates = np.array(coordinates)
                        #line_name  = os.path.splitext(listOfiles[iter].filename)
                        #line_name =line_name[0] #just use the former one 
                        #pathy = rois[line_name]['y']
                        #pathx = rois[line_name]['x']

                        # delete the coordinate out side the boundary of image
                        len_ori,_ = coordinates.shape
                        target =0
                        for j in range(len_ori):

                            if (coordinates[target,0] < 0 or coordinates[target,0]>(W-1) ) :# check the x coordinates
                                coordinates= np.delete(coordinates , target,axis=0)
                            else:
                                target +=1


                        pathy  = coordinates[:,1]
                        pathx  = coordinates[:,0]

                        num_points = len(pathx)
                        path_w  = pathx[num_points-1] - pathx[0]
                        path_w = int(path_w)
                        # sometimes the contour is plot in reversed direction  
                        if  path_w<0:
                            path_w=-path_w
                            pathy=pathy[::-1]
                            pathx=pathx[::-1]

                        pathyl =  np.ones(int(path_w)) * np.nan

                        for j in range (num_points):
                            #importante sometimes the start point is nnot the lestmost
                             this_index = np.clip(  pathx[j] - pathx[0], 0,path_w-1)
                             pathyl[int(this_index)] = float (pathy[j] -1)
                        add_3   = np.append(pathyl[::-1],pathyl,axis=0) # cascade
                        add_3   = np.append(add_3,pathyl[::-1],axis=0) # cascade
                        s = pd.Series(add_3)
                        pathyl = s.interpolate(method = "linear")
                        pathyl = pathyl[path_w:2*path_w].to_numpy() 
                        pathxl = np.arange(int(pathx[0]) , int(pathx[0]) + path_w )
                       


                        if len(pathxl) > 0.96 *W: #  correct  the unperfect label
            # rememver to add resacle later
                            pathyl = signal.resample(pathyl, W)
                            pathxl = np.arange(0, W)
                            

                        contoursx[iter] = pathxl
                        contoursy[iter] = pathyl
                        #max_buffer[iter] = np.min(pathyl)  #  use the minimal value to detemine which contour as the 1st and 2nd .....
                        max_buffer[iter] = np.max(pathyl)  #  use the mean value to detemine which contour as the 1st and 2nd .....
                        
                        pass
                    

                     #  use the minimal value to detemine which contour as the 1st and 2nd .....
                    new_index  = np.argsort( max_buffer)
                    self.contours_x = [None]*len_list
                    self.contours_y = [None]*len_list

                    for iter in range(len_list):
                        self.contours_x[iter] = contoursx[new_index[iter]]
                        self.contours_y[iter] = contoursy[new_index[iter]]
                        if self.display_flag == True:
                            img1 = self.draw_coordinates_color(img1,self.contours_x[iter],
                                                               self.contours_y[iter],iter)

                    #save this result 
                    self.img_num = a
                    #self.contours_x = [path0ln, path1ln, path2ln, path3ln]
                    #self.saver.append_new_name_contour (self.img_num,self.contours,self.database_root)
                    self.saver.append_new_name_contour (self.img_num,self.contours_x,self.contours_y,self.save_dir)

                    cv2.imshow('pic',img1)
                    print(str(a))
                    cv2.waitKey(10) 

if __name__ == '__main__':
    #play with JSON gile 
    #file_dir   = "D:/Deep learning/dataset/label data/label/0.json"

    #with open(file_dir) as f:
    #    data = JSON.load(f)
    ## Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
    #print(data)
    #newdata   = data 
    #shape  = data["shapes"]
    #print(shape)
    #num_line  = len(shape)
    #coordinate  = shape[0]["points"]
    #num_points  = len(coordinate)
    #for  i in range(num_points):
    #    coordinate[i][1] = 100
    #print(coordinate)
    ### modify the coordinate 
    #newdata["shapes"][0]["points"]  = coordinate
    ##save 
    #with open(file_dir, "w") as jsonFile:
    #    JSON.dump(newdata, jsonFile)


    converter  = Read_read_check_json_label()
    converter.check_one_folder() # convert the Json file into pkl files 


