from model import cGAN_build2
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
from scipy import signal 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Pair2Path(object):
    def __init__(self ):
        pth_save_dir  = "../../DeepLearningModel/pair2path/cGANG_epoch_4.pth"
        creator = cGAN_build2.CGAN_creator() # the Cgan for the segmentation 
        self.GANmodel= creator.creat_cgan()  #  G and D are created here 
        #netD = gan_body._netD_Resnet()
        #self.GANmodel.netG.apply(weights_init)
        self.GANmodel.netG.cuda()
         
   
        self.Resample_size =256
        
        self. Original_window_Len  = 71
 
        #self.netD.eval()
        torch.no_grad()

        #my_netD  = ShiftingNetBody.ShiftingNet_init_my( None)

 
        self.GANmodel.netG.load_state_dict(torch.load(pth_save_dir))
        #self. netD.load_state_dict(torch.load(dir_netD))
        print(self.GANmodel.netG)

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
    def predict(self,img1,img2):
        #multi_scale_weight = [0.005, 0.01, 0.02, 0.16, 0.32]
        #multi_scale_weight = [0.2, 0.2, 0.2, 0.2, 0.2]


        H,W = img1.shape
        pair1  =   img1 
        pair2  =   img2 
        #pair1  =   img1[self.Crop_end:H,:] 
        #pair2  =   img3[self.Crop_end:H,:] 
        #pair3  =   img1
        #pair4  =   img2
        #pair3  =   img2
        #pair4  =   img1
        #pair3  =   pair1
        #pair4  =   pair2
        #pair4  =   pair2
        pair1  =  cv2.resize(pair1, (self.Resample_size,self.Resample_size), interpolation=cv2.INTER_AREA)   -104.0
        pair2  =  cv2.resize(pair2, (self.Resample_size,self.Resample_size), interpolation=cv2.INTER_AREA)   -104.0
        
        #pair1  =  cv2.resize(self.image2_append(pair1), (self.Resample_size,self.Resample_size2), interpolation=cv2.INTER_AREA)   -104.0
        #pair2  =  cv2.resize(self.image2_append(pair2), (self.Resample_size,self.Resample_size2), interpolation=cv2.INTER_AREA)   -104.0
        #pair3  =  cv2.resize(self.image2_append(pair3), (self.Resample_size,self.Resample_size2), interpolation=cv2.INTER_AREA)   -104.0
        #pair4  =  cv2.resize(self.image2_append(pair4), (self.Resample_size,self.Resample_size2), interpolation=cv2.INTER_AREA)   -104.0
        np_input = numpy.zeros((1,2,self.Resample_size,self.Resample_size)) # a batch with piece num
        np_input[0,0,:,:] = pair1
        np_input[0,1,:,:] = pair2
        #np_input[0,2,:,:] = pair3
        #np_input[0,3,:,:] = pair4

        input = torch.from_numpy(numpy.float32(np_input)) 
        input = input.to(device)
        #inputv = Variable(input.unsqueeze(0))
        inputv = Variable(input)
        self.GANmodel.set_G_input(  inputv)         # unpack data from dataset and apply preprocessing
        self.GANmodel.forward()

        output = self.GANmodel.out_pathes0
        save_out  = output
        save_out = save_out[0] 
        save_out = save_out[0] 
        save_out_long = save_out 
        save_out_long = save_out_long.cpu().detach().numpy()


        save_out_long = save_out_long * self.Original_window_Len
        save_out_long  = signal.resample(save_out_long, W)
        
        return  save_out_long

if __name__ == '__main__': 
    # test
    predictor =  Pair2Path()