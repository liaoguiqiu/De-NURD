from analy import MY_ANALYSIS
from analy import Save_signal_enum
import cv2
import numpy as np
from scipy import signal
import os
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
import gan_body
from arg_parse import opt
import re
from image_trans import BaseTransform 
from scipy.ndimage import gaussian_filter1d
from time  import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

opt.netD  = "..\\DeepPathFinding\\out\\netD_epoch_13.pth"
generatepath = "..\\DeepPathFinding\\generated\\"
save_test_dir = "..\\DeepPathFinding\\save_test\\"
save_compare_dir = "..\\DeepPathFinding\\save_compare\\"

Matrix_dir =  "..\\dataset\\CostMatrix\\2\\"
#Matrix_dir =  "..\\dataset\\DCGAN data\\un_processed_mat\\"
from dataset import myDataloader,Batch_size,Resample_size, Path_length

saved_stastics = MY_ANALYSIS()
saved_stastics=saved_stastics.read_my_signal_results()
saved_stastics.display()
saved_stastics.display()
read_id =0
#transform = BaseTransform(  64,(104/256.0, 117/256.0, 123/256.0))
transform = BaseTransform(  Resample_size,[104])  #gray scale data

#netD = gan_body._netD_8()
netD = gan_body._netD_Resnet()


 
print('load weights for Path_ find ing')
     
netD.load_state_dict(torch.load(opt.netD))
print(netD)
netD.cuda()
netD.eval()
while (1):
    
    start_time = time()
    #get the Id of image which should be poibnt to
    Image_ID = int( saved_stastics.signals[Save_signal_enum.image_iD.value, read_id])
    
    #get the path
    path  = saved_stastics.path_saving[read_id,:]
    #check whether the ID of image point to the path or not
    img_path1 = Matrix_dir + str(Image_ID)+ ".jpg"
    Image = cv2.imread(img_path1)
    gray  =   cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)   
    H,W= gray.shape  #get size of image

    #resample 
    piece_num = 10 # the number of the squers 
    piece_W = W/piece_num
    input_batch = np.zeros((piece_num,3,Resample_size,Resample_size)) # a batch with piece num

    for  slice_point in range (piece_num):
        star_pt = int(slice_point*piece_W)
        end_pt = int ((slice_point+1)*piece_W)
        img_piece = gray[:,star_pt:end_pt]
        img_piece = cv2.resize(img_piece, (Resample_size,Resample_size), interpolation=cv2.INTER_AREA)
        input_batch[slice_point,0,:,:] = transform(img_piece)[0]
        input_batch[slice_point,1,:,:] = transform(img_piece)[0]
        input_batch[slice_point,2,:,:] = transform(img_piece)[0]
    input = torch.from_numpy(np.float32(input_batch)) 
    input = input.to(device) 


    #inputv = Variable(input.unsqueeze(0))
    inputv = Variable(input )

    output = netD(inputv)
    #output = output.view(Batch_size,Resample_size).squeeze(1)
 

    #gray2  =   cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
   
    output = output.cpu().detach().numpy()
    #long_out  = np.append(output,output)
    #long_out  = np.append(long_out,output)
    path_upsam = np.zeros(W)
    connected_path = []
    for connect_point in range (piece_num):
         connected_path= np.append(connected_path,output[connect_point,:])
             
    path_upsam = signal.resample(connected_path,W)*H
    path_upsam = gaussian_filter1d (path_upsam ,3)
        #cv2.imshow('step_process',gray_video1)  

    #gray1  =   cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)   
    show1 = gray.astype(float)
    
    for i in range ( len(path_upsam)):
        show1[int(path_upsam[i]),i]=254
    cv2.imshow('Deeplearning full',show1.astype(np.uint8)) 
    cv2.imwrite(save_test_dir  + str(Image_ID) +".jpg", show1)

        #cv2.imshow('step_process',gray_video1)  
    #reconstruct the path 
    show3 = gray.astype(float)
    
    for i in range ( len(path)):
        show3[int(path[i]),i]=254
    cv2.imshow('original full',show3.astype(np.uint8)) 
    cv2.imwrite(save_compare_dir  + str(Image_ID) +".jpg", show3)

        #cv2.imshow('step_process',gray_video1)  

    read_id+=1

    if cv2.waitKey(12) & 0xFF == ord('q'):
        break
    end_time=time()
    print ('function vers1 takes %f' %(end_time-start_time))



 
     




