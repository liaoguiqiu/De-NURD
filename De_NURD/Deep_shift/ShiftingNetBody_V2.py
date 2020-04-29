import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
import torch.nn.functional as F

 
#   conv with default kernel 3*3
#with inpuut plane and out put plane can be modified 
def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )
 
# with default out plane 2 
# for our use  it can be 1   
def predict_shift(in_planes):
    #LGQ
    return nn.Conv2d(in_planes, 1,kernel_size=3,stride=1,padding=1,bias=False)
# basic de-conv layer
def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )

# crop the edge ,  change the size of input
def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]

    
# the shift network
class _ShiftingNet(nn.Module):
    expansion = 1

    def __init__(self,batchNorm=True):
        super(_ShiftingNet,self).__init__()

        self.batchNorm = batchNorm
        #LGQ the input is change into gray scale
        self.conv1   = conv(self.batchNorm,   4,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026-1,256)
        self.deconv3 = deconv(770-1,128)
        self.deconv2 = deconv(386-1,64)

        #LGQ cancat size is modified  size = size -1 
        # due to our is a single v
        self.predict_shift6 = predict_shift(1024)
        self.predict_shift5 = predict_shift(1026-1)
        self.predict_shift4 = predict_shift(770-1)
        self.predict_shift3 = predict_shift(386-1)
        self.predict_shift2 = predict_shift(194-1)

    
        # LGQ change the array into 1 D
        self.upsampled_shift6_to_5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_shift5_to_4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_shift4_to_3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_shift3_to_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)

     

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        shift6       = self.predict_shift6(out_conv6)
        shift6_up    = crop_like(self.upsampled_shift6_to_5(shift6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5,out_deconv5,shift6_up),1)
        shift5       = self.predict_shift5(concat5)
        shift5_up    = crop_like(self.upsampled_shift5_to_4(shift5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4,out_deconv4,shift5_up),1)
        shift4       = self.predict_shift4(concat4)
        shift4_up    = crop_like(self.upsampled_shift4_to_3(shift4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,shift4_up),1)
        shift3       = self.predict_shift3(concat3)
        shift3_up    = crop_like(self.upsampled_shift3_to_2(shift3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2,out_deconv2,shift3_up),1)
        shift2 = self.predict_shift2(concat2)

        if self.training:
            return shift2,shift3,shift4,shift5,shift6
        else:
            return shift2,shift3,shift4,shift5,shift6

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

     

def ShiftingNet_init(data=None):
 
    #Args:
    #    data : pretrained weights of the network. will create a new one if not set
    #"""
    model = _ShiftingNet(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
  
def ShiftingNet_bn_init(data=None):
    #Args:
    #    data : pretrained weights of the network. will create a new one if not set
    #"""
    model = _ShiftingNet(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
