from Deep_shift import ShiftingNetBody_V2
#import ShiftingNetBody_V2

 
import cv2
import numpy  
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
class Shift_Predict(object):
    def __init__(self ):
        dir_netD  = "../../DeepLearningModel/shift/netD_epoch_44.pth"

        self.Crop_start = 0
        self.Crop_end  = 200
        self.Resample_size =128
        self.Resample_size2 =128
        self. Original_window_Len  = 71
        self.netD = ShiftingNetBody_V2.ShiftingNet_init( None)
        self.netD.cuda()
        self.netD.eval()
        #my_netD  = ShiftingNetBody.ShiftingNet_init_my( None)

 
 
        self. netD.load_state_dict(torch.load(dir_netD))
        print(self. netD)

        pass


    # the image sequence 1: new  2the stablized history 3: the reference
    def predict(self,img1,img2,img3):
        H,W = img1.shape
        pair1  =   img1[self.Crop_start:self.Crop_end,:] 
        pair2  =   img3[self.Crop_start:self.Crop_end,:] 
        pair3  =   img1
        pair4  =   img2
        pair1  =  cv2.resize(pair1, (self.Resample_size,self.Resample_size2), interpolation=cv2.INTER_AREA)   -104.0
        pair2  =  cv2.resize(pair2, (self.Resample_size,self.Resample_size2), interpolation=cv2.INTER_AREA)   -104.0
        pair3  =  cv2.resize(pair3, (self.Resample_size,self.Resample_size2), interpolation=cv2.INTER_AREA)   -104.0
        pair4  =  cv2.resize(pair4, (self.Resample_size,self.Resample_size2), interpolation=cv2.INTER_AREA)   -104.0
        np_input = numpy.zeros((1,4,self.Resample_size,self.Resample_size2)) # a batch with piece num
        np_input[0,0,:,:] = pair1
        np_input[0,1,:,:] = pair2
        np_input[0,2,:,:] = pair3
        np_input[0,3,:,:] = pair4

        input = torch.from_numpy(numpy.float32(np_input)) 
        input = input.to(device)
        #inputv = Variable(input.unsqueeze(0))
        inputv = Variable(input)
        output = self.netD(inputv)
        save_out  = output
        save_out = save_out[4] 
        save_out = save_out[0] 


        save_out  = (save_out.data.mean()) *(self.Original_window_Len   ) 
        save_out  = numpy.clip(int(save_out),0, self.Original_window_Len   -1)
        return  save_out



