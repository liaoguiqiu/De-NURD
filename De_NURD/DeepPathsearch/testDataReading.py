from analy import MY_ANALYSIS
from analy import Save_signal_enum
import cv2
import numpy as np
from scipy import signal
Matrix_dir =  "..\\dataset\\CostMatrix\\1\\"



saved_stastics = MY_ANALYSIS()
saved_stastics=saved_stastics.read_my_signal_results()
saved_stastics.display()
saved_stastics.display()
read_id =0
Resample_size  = 320
while (1):
    
    #get the Id of image which should be poibnt to
    Image_ID = int( saved_stastics.signals[Save_signal_enum.image_iD.value, read_id])
    #get the path
    path  = saved_stastics.path_saving[read_id,:]

    # processed
    #check whether the ID of image point to the path or not
    img_path1 = Matrix_dir + str(Image_ID)+ ".jpg"
    Image = cv2.imread(img_path1)
    gray  =   cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    show1 =  gray.astype(float)
     
    for i in range ( len(path)):
        show1[int(path[i]),i]=254
    cv2.imshow('step_process',show1.astype(np.uint8)) 

    #show the reampled version
    img2 = cv2.resize(Image, (Resample_size,Resample_size), interpolation=cv2.INTER_AREA)
    gray2  =   cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    path2 =  signal.resample(path, Resample_size)
    show2 =  gray2.astype(float)
    path2=path2/71.0*Resample_size
    for i in range ( len(path2)):
        show2[int(path2[i]),i]=254
    cv2.imshow('step_process2',show2.astype(np.uint8)) 

    #cv2.imshow('step_process',gray_video1)  
    read_id+=1
    if cv2.waitKey(12) & 0xFF == ord('q'):
        break




