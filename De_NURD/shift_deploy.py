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
        dir_netD  = "../../DeepLearningModel/shift/netD_epoch_2.pth"

        self.Crop_start = 80
        self.Crop_end  = 200
        self.Resample_size =512
        self.Resample_size2 =200
        self. Original_window_Len  = 71
        self.netD = ShiftingNetBody_V2.ShiftingNet_init( None)
        self.netD.cuda()
        self.netD.eval()
        #my_netD  = ShiftingNetBody.ShiftingNet_init_my( None)

 
 
        self. netD.load_state_dict(torch.load(dir_netD))
        print(self. netD)

        pass
    def image3_append(self,img):
        long = numpy.append(img,img,axis=1)
        long = numpy.append(long,img,axis=1)
        return long
    def image2_append(self,img):
        long = numpy.append(img,img,axis=1)
        #long = np.append(long,img,axis=1)
        return long
    # the image sequence 1: new  2the stablized history 3: the reference
    def predict(self,img1,img2,img3):
        multi_scale_weight = [0.005, 0.01, 0.02, 0.16, 0.32]

        H,W = img1.shape
        pair1  =   img1[self.Crop_start:self.Crop_end,:] 
        pair2  =   img3[self.Crop_start:self.Crop_end,:] 
        pair3  =   img1
        pair4  =   img2
        #pair4  =   pair2
        pair1  =  cv2.resize(pair1, (self.Resample_size,self.Resample_size2), interpolation=cv2.INTER_AREA)   -104.0
        pair2  =  cv2.resize(pair2, (self.Resample_size,self.Resample_size2), interpolation=cv2.INTER_AREA)   -104.0
        pair3  =  cv2.resize(pair3, (self.Resample_size,self.Resample_size2), interpolation=cv2.INTER_AREA)   -104.0
        pair4  =  cv2.resize(pair4, (self.Resample_size,self.Resample_size2), interpolation=cv2.INTER_AREA)   -104.0
        
        #pair1  =  cv2.resize(self.image2_append(pair1), (self.Resample_size,self.Resample_size2), interpolation=cv2.INTER_AREA)   -104.0
        #pair2  =  cv2.resize(self.image2_append(pair2), (self.Resample_size,self.Resample_size2), interpolation=cv2.INTER_AREA)   -104.0
        #pair3  =  cv2.resize(self.image2_append(pair3), (self.Resample_size,self.Resample_size2), interpolation=cv2.INTER_AREA)   -104.0
        #pair4  =  cv2.resize(self.image2_append(pair4), (self.Resample_size,self.Resample_size2), interpolation=cv2.INTER_AREA)   -104.0
        np_input = numpy.zeros((1,4,self.Resample_size2,self.Resample_size)) # a batch with piece num
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
        ave_out =0
        for k in range(len(multi_scale_weight)):
            this_out =      output[k] 
            this_out = this_out[0] 
            this_out  = (this_out.data.mean()) *(self.Original_window_Len )  
            ave_out += this_out*multi_scale_weight[k]

        ave_out /= numpy.sum(multi_scale_weight)

        save_out  = (save_out.data.mean()) *(self.Original_window_Len   ) 
        save_out = ave_out
        save_out  = numpy.clip(int(save_out),0, self.Original_window_Len   -1)
        return  save_out



