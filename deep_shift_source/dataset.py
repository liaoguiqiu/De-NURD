import cv2
import numpy as np
import os
from analy import MY_ANALYSIS
from analy import Save_signal_enum
from scipy import signal 
from image_trans import BaseTransform  
from random import seed
from random import random
seed(1)
Batch_size = 100
Resample_size =80
Resample_size2 = 80
Path_length = 20
Mat_size   = 80
 
transform_img = BaseTransform(  Resample_size,[104])  #gray scale data
transform_mat = BaseTransform(  Mat_size,[104])  #gray scale data


class myDataloader(object):
    def __init__(self, batch_size,image_size,path_size):
        self.data_pair1_root = "../dataset/For_pair_IMG_Train/pair1/"
        self.data_pair2_root = "..\\dataset\\For_pair_IMG_Train\\pair2\\"
        self.data_mat_root = "../dataset/For_pair_IMG_Train/CostMatrix/"
        self.signalroot ="..\\dataset\\For_pair_IMG_Train\\saved_stastics\\" 
        self.read_all_flag=0
        self.read_record =0
        self.folder_pointer = 0
        self.batch_size  = batch_size
        self.img_size  = Resample_size
        self.path_size  = Path_length
        self.mat_size  = Mat_size
        self.img_size2  = Resample_size2


        # Initialize the inout for the tainning
        self.input_mat = np.zeros((batch_size,1,Mat_size,Mat_size)) #matri
        self.input_path = np.zeros((batch_size,Path_length))#path
        self.input_pair1 = np.zeros((batch_size,1,Resample_size2,Resample_size))#pairs
        self.input_pair2 = np.zeros((batch_size,1,Resample_size2,Resample_size))
        self.input_pair3 = np.zeros((batch_size,1,Resample_size2,Resample_size))

        # the number isdeter by teh mat num
        self.all_dir_list = os.listdir(self.data_mat_root)
        self.folder_num = len(self.all_dir_list)
        # create the buffer list(the skill to create the list)
        self.folder_mat_list = [None]*self.folder_num
        self.folder_pair1_list = [None]*self.folder_num
        self.folder_pair2_list = [None]*self.folder_num
        self.signal = [None]*self.folder_num

        # create all  the folder list and their data list

        number_i = 0
        # all_dir_list is subfolder list 
        #creat the image list point to the STASTICS TIS  list
        saved_stastics = MY_ANALYSIS()
        #read all the folder list of mat and pairs and path
        for subfold in self.all_dir_list:
            #the mat list
            this_folder_list =  os.listdir(os.path.join(self.data_mat_root, subfold))
            this_folder_list2 = [ self.data_mat_root +subfold + "\\" + pointer for pointer in this_folder_list]
            self.folder_mat_list[number_i] = this_folder_list2

            #the pair1 list
            this_folder_list =  os.listdir(os.path.join(self.data_pair1_root, subfold))
            this_folder_list2 = [ self.data_pair1_root +subfold + "\\" + pointer for pointer in this_folder_list]
            self.folder_pair1_list[number_i] = this_folder_list2
            #the pair2 list
            this_folder_list =  os.listdir(os.path.join(self.data_pair2_root, subfold))
            this_folder_list2 = [ self.data_pair2_root +subfold + "\\" + pointer for pointer in this_folder_list]
            self.folder_pair2_list[number_i] = this_folder_list2
            #the supervision signal list
               #change the dir firstly before read
            saved_stastics.all_statics_dir = os.path.join(self.signalroot, subfold, 'signals.pkl')
            self.signal[number_i]  =  saved_stastics.read_my_signal_results()
            
            number_i +=1
            #read the folder list finished  get the folder list and all saved path
    def gray_scale_augmentation(self,orig_gray,amplify_value) :
        random_scale = 0.7 + (1.5  - 0.7) *amplify_value
        aug_gray = orig_gray * random_scale
        aug_gray = np.clip(aug_gray, a_min = 1, a_max = 254)
        return aug_gray

    # let read a bathch
    def read_a_batch(self):
        read_start = self.read_record
        #read_end  = self.read_record+ self.batch_size
        thisfolder_len =  len (self.folder_mat_list[self.folder_pointer])
        
            #return self.input_mat,self.input_path# if out this folder boundary, just returen
        this_pointer=0
        i=read_start
        while (1):
        #for i in range(read_start, read_end):
            #this_pointer = i -read_start
            # get the all the pointers 
            #Image_ID , b = os.path.splitext(os.path.dirname(self.folder_list[self.folder_pointer][i]))
            Path_dir,Image_ID =os.path.split(self.folder_mat_list[self.folder_pointer][i])
            Image_ID_str,jpg = os.path.splitext(Image_ID)
            Image_ID = int(Image_ID_str)
            #start to read image and paths to fill in the input bach
            this_mat_path = self.folder_mat_list[self.folder_pointer][i] # read saved mat
            this_mat = cv2.imread(this_mat_path)
            this_pair1_path = self.folder_pair1_list[self.folder_pointer][i] # read saved pair1
            this_pair1 = cv2.imread(this_pair1_path)
            this_pair2_path = self.folder_pair2_list[self.folder_pointer][i] # read saved pair1
            this_pair2 = cv2.imread(this_pair2_path)
            #resample 
            #this_img = cv2.resize(this_img, (self.img_size,self.img_size), interpolation=cv2.INTER_AREA)
           
            #get the index of this Imag path
            Path_Index_list = self.signal[self.folder_pointer].signals[Save_signal_enum.image_iD.value,:]
            Path_Index_list = Path_Index_list.astype(int)
            Path_Index_list = Path_Index_list.astype(str)

            try:
                Path_Index = Path_Index_list.tolist().index(Image_ID_str)
            except ValueError:
                print(Image_ID_str + "not path exsting")

            else:             
                Path_Index = Path_Index_list.tolist().index(Image_ID_str)            
                this_path = self.signal[self.folder_pointer].path_saving[Path_Index]
                #path2 =  signal.resample(this_path, self.path_size)#resample the path
                # concreate the image batch and path
                this_mat  =   cv2.cvtColor(this_mat, cv2.COLOR_BGR2GRAY)
                this_pair1  =   cv2.cvtColor(this_pair1, cv2.COLOR_BGR2GRAY)
                this_pair2  =   cv2.cvtColor(this_pair2, cv2.COLOR_BGR2GRAY)
                                    #data augmentation

                
                # imag augmentation
                amplifier  = random()
                this_mat = self.gray_scale_augmentation(this_mat,amplifier)
                H_mat,W_mat = this_mat.shape
                H_img,W_img = this_pair1.shape
                piece_num  = int(W_img/self.img_size)
                piece_W  = self.img_size
                slides_step = int(piece_W/5)
                slice_point = 0
                # fill the batch with small pieces
                while(1):
                    start_point = slice_point * slides_step
                    end_point  = start_point + piece_W
                    if (end_point>=W_img):
                        break
                    slice_point +=1
                    #cut 
                    mat_piece = this_mat[:,start_point:end_point]
                    pair1_piece = this_pair1[:,start_point:end_point]
                    pair2_piece = this_pair2[:,start_point:end_point]
                    middle  =int( (start_point+end_point)/2)
                    path_piece = this_path[middle-int(Path_length/2):middle+int(Path_length/2)]
                    #resample to qure
                    mat_piece = cv2.resize(mat_piece, (self.mat_size,self.mat_size), interpolation=cv2.INTER_AREA)
                    pair1_piece = cv2.resize(pair1_piece, (self.img_size,self.img_size2), interpolation=cv2.INTER_AREA)
                    pair2_piece = cv2.resize(pair2_piece, (self.img_size,self.img_size2), interpolation=cv2.INTER_AREA)      
                    path_piece =  signal.resample(path_piece, self.path_size)#resample the path
                    #data augmentation
                    amplifier  = random()
                    pair1_piece =   self.gray_scale_augmentation(pair1_piece,amplifier)
                    pair2_piece =   self.gray_scale_augmentation(pair2_piece,amplifier)
                    
 
                    self.input_mat[this_pointer,0,:,:] = transform_mat(mat_piece)[0]
                    #self.input_pair1[this_pointer,0,:,:] = transform_img(pair1_piece)[0]
                    #self.input_pair2[this_pointer,0,:,:] = transform_img(pair2_piece)[0]
                    self.input_pair1[this_pointer,0,:,:] = pair1_piece -104
                    self.input_pair2[this_pointer,0,:,:] = pair2_piece - 104
                    self.input_path [this_pointer , :] = (path_piece)/(H_mat)
                    this_pointer +=1
                    if(this_pointer>=self.batch_size): # this batch has been filled
                         break


            i+=1
            if (i>=thisfolder_len):
                i=0
                self.read_record =0
                self.folder_pointer+=1
                if (self.folder_pointer>= self.folder_num):
                    self.read_all_flag =1
                    self.folder_pointer =0
            if(this_pointer>=self.batch_size): # this batch has been filled
                break
            pass
        self.read_record=i # after reading , remember to  increase it 
        return self.input_mat,self.input_path


##test read 
#data  = myDataloader (Batch_size,Resample_size,Path_length)

#for  epoch in range(500):

#    while(1):
#        data.read_a_batch()


