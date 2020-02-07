import cv2
import numpy as np
import os
from analy import MY_ANALYSIS
from analy import Save_signal_enum
from scipy import signal 
from image_trans import BaseTransform  
Batch_size = 128
Resample_size =128
Path_length = 128
 
 
transform = BaseTransform(  Resample_size,[104])  #gray scale data

class myDataloader(object):
    def __init__(self, batch_size,image_size,path_size):
        self.dataroot = "..\\dataset\\CostMatrix\\"
        self.signalroot ="..\\dataset\\saved_stastics\\" 
        self.read_all_flag=0
        self.read_record =0
        self.folder_pointer = 0
        self.batch_size  = batch_size
        self.img_size  = image_size
        self.path_size  = path_size


        self.input_image = np.zeros((batch_size,1,image_size,image_size))
        self.input_path = np.zeros((batch_size,path_size))
        self.all_dir_list = os.listdir(self.dataroot)
        self.folder_num = len(self.all_dir_list)
        # create the buffer list
        self.folder_list = [None]*self.folder_num
        self.signal = [None]*self.folder_num

        # create all  the folder list and their data list

        number_i = 0
        # all_dir_list is subfolder list 
        #creat the image list point to the STASTICS TIS  list
        saved_stastics = MY_ANALYSIS()
        #read all the folder list
        for subfold in self.all_dir_list:
            #if(number_i==0):
            this_folder_list =  os.listdir(os.path.join(self.dataroot, subfold))
            this_folder_list2 = [ self.dataroot +subfold + "\\" + pointer for pointer in this_folder_list]
            self.folder_list[number_i] = this_folder_list2

            #change the dir firstly before read
            saved_stastics.all_statics_dir = os.path.join(self.signalroot, subfold, 'signals.pkl')
            self.signal[number_i]  =  saved_stastics.read_my_signal_results()
            number_i +=1
            #read the folder list finished  get the folder list and all saved path

    def read_a_batch(self):
        read_start = self.read_record
        #read_end  = self.read_record+ self.batch_size
        thisfolder_len =  len (self.folder_list[self.folder_pointer])
        
            #return self.input_image,self.input_path# if out this folder boundary, just returen
        this_pointer=0
        i=read_start
        while (1):
        #for i in range(read_start, read_end):
            #this_pointer = i -read_start
            # get the all the pointers 
            #Image_ID , b = os.path.splitext(os.path.dirname(self.folder_list[self.folder_pointer][i]))
            Path_dir,Image_ID =os.path.split(self.folder_list[self.folder_pointer][i])
            Image_ID_str,jpg = os.path.splitext(Image_ID)
            Image_ID = int(Image_ID_str)
            #start to read image and paths to fill in the input bach
            this_image_path = self.folder_list[self.folder_pointer][i] # read saved path
            this_img = cv2.imread(this_image_path)
            #resample 
            this_img = cv2.resize(this_img, (self.img_size,self.img_size), interpolation=cv2.INTER_AREA)
           
            #get the index of this Imag path
            Path_Index_list = self.signal[self.folder_pointer].signals[Save_signal_enum.image_iD.value,:]
            Path_Index_list = Path_Index_list.astype(int)
            Path_Index_list = Path_Index_list.astype(str)

            try:
                Path_Index = Path_Index_list.tolist().index(Image_ID_str)
            except ValueError:
                print(Image_ID_str + "not path exsting")
                i+=1
                
            else:             
                Path_Index = Path_Index_list.tolist().index(Image_ID_str)            
                this_path = self.signal[self.folder_pointer].path_saving[Path_Index]
                path2 =  signal.resample(this_path, self.path_size)#resample the path
                # concreate the image batch and path
                this_gray  =   cv2.cvtColor(this_img, cv2.COLOR_BGR2GRAY)
                self.input_image[this_pointer,0,:,:] = transform(this_gray)[0]
                self.input_path [this_pointer , :] = path2
                this_pointer +=1
                i+=1
            if (i>=thisfolder_len):
                i=0
                self.read_record =0
                self.folder_pointer+=1
                if (self.folder_pointer>= self.folder_num):
                    self.read_all_flag =1
                    self.folder_pointer =0
            if(this_pointer>=self.batch_size):
                break
            pass
        self.read_record=i # after reading , remember to  increase it 
        return self.input_image,self.input_path


##test read 
#data  = myDataloader (Batch_size,Resample_size,Path_length)

#for  epoch in range(500):

#    while(1):
#        data.read_a_batch()


