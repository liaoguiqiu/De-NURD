import torch
import torch.nn as nn
import arg_parse
from arg_parse import kernels, strides, pads
from dataset_shifting import Path_length,Batch_size
import torchvision.models
nz = int(arg_parse.opt.nz)
ngf = int(arg_parse.opt.ngf)
ndf = int(arg_parse.opt.ndf)
nc = 2


class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 16, kernels[0], strides[0], pads[0], bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernels[1], strides[1], pads[1], bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernels[2], strides[2], pads[2], bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 4,     ngf*2, kernels[3], strides[3], pads[3], bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2,     ngf, kernels[4], strides[4], pads[4], bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(    ngf,      nc, kernels[5], strides[5], pads[5], bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
           
        )

    def forward(self, input):
        output = self.main(input)
        return output


class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()

        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
         # input is (nc) x 128 x 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(nc, ndf, kernels[5], strides[5], pads[5], bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # input is (nc) x 64 x 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernels[4], strides[4], pads[4], bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # state size. (ndf) x 32 x 32
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf * 4, kernels[3], strides[3], pads[3], bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # state size. (ndf*2) x 16 x 16
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernels[2], strides[2], pads[2], bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # state size. (ndf*4) x 8 x 8
        self.conv5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 16, kernels[1], strides[1], pads[1], bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # state size. (ndf*8) x 4 x 4
        self.conv6 = nn.Sequential(
            nn.Conv2d(ndf * 16, Path_length, kernels[0], strides[0], pads[0], bias=False),
            nn.Sigmoid()
        )
        #self.conv6 = nn.Sequential(
        #    nn.Conv2d(ndf * 16, 1, kernels[0], strides[0], pads[0], bias=False),
        #    nn.Sigmoid()
        #)


    def forward(self, input):
        # output = self.main(input)

        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        #return out_conv6.view(-1, 1).squeeze(1)
        return out_conv6 

    def get_features(self, input):
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)

        max_pool1 = nn.MaxPool2d(int(out_conv1.size(2) / 4))
        max_pool2 = nn.MaxPool2d(int(out_conv2.size(2) / 4))
        max_pool3 = nn.MaxPool2d(int(out_conv3.size(2) / 4))
        # max_pool4 = nn.MaxPool2d(int(out_conv4.size(2) / 4))

        vector1 = max_pool1(out_conv1).view(input.size(0), -1).squeeze(1)
        vector2 = max_pool2(out_conv2).view(input.size(0), -1).squeeze(1)
        vector3 = max_pool3(out_conv3).view(input.size(0), -1).squeeze(1)
        # vector4 = max_pool4(out_conv4).view(input.size(0), -1).squeeze(1)

        return torch.cat((vector1, vector2, vector3), 1)

    
class _netD_8(nn.Module):
    def __init__(self):
        super(_netD_8, self).__init__()

        layer_len = len(kernels)
        #create the layer list
        self.layers = nn.ModuleList()
        for layer_pointer in range(layer_len):
             # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
             # input is (nc) x 128 x 128
            if  layer_pointer ==0:
                this_input_depth = nc
                this_output_depth = ndf
            else:
                this_input_depth = this_output_depth
                this_output_depth = this_input_depth*2


    
            if (layer_pointer == (layer_len-1)):
                #self.layers = nn.Sequential(
                #nn.Conv2d(this_input_depth, Path_length, kernels[layer_len -layer_pointer-1], strides[layer_len -layer_pointer-1], pads[layer_len -layer_pointer-1], bias=False),
                #nn.Sigmoid()
                # )
                self.layers.append(
                nn.Conv2d(this_input_depth, Path_length, kernels[layer_len -layer_pointer-1], strides[layer_len -layer_pointer-1], pads[layer_len -layer_pointer-1], bias=False),          
                 )
                #self.layers.append (
                #nn.BatchNorm2d(Path_length),
                #   )
                self.layers.append(
                nn.Sigmoid()
                 )
            else:
                  # input is (nc) x 64 x 64
                self.layers.append (
                nn.Conv2d(this_input_depth, this_output_depth, kernels[layer_len -layer_pointer-1], strides[layer_len -layer_pointer-1], pads[layer_len -layer_pointer-1], bias=False),
                )
                self.layers.append (
                nn.BatchNorm2d(this_output_depth),
                   )
                self.layers.append (
                nn.LeakyReLU(0.2, inplace=True)
                )
            #self.layers.append(this_layer)
    def forward(self, x):
        #output = self.main(input)
        #layer_len = len(kernels)
        #for layer_point in range(layer_len):
        #    if(layer_len==0):
        #        output = self.layers[layer_point](input)
        #    else:
        #        output = self.layers[layer_point](output)
        for i, name in enumerate(self.layers):
            x = self.layers[i](x)


        return x 


# mainly based on the resnet     
class _netD_Resnet2(nn.Module):
    def __init__(self):
        super(_netD_Resnet, self).__init__()

        #layer_len = len(kernels)
        #create the layer list
        self.layers = nn.ModuleList()
        #self.resnet18 = torchvision.models.resnet18(pretrained = False, **kwargs)
        self.resnet18 = torchvision.models.resnet18(pretrained = True)
        #self.resnet34 = torchvision.models.resnet34(pretrained = False)
        #self.resnet101 = torchvision.models.resnet101(pretrained = False)



        self.layers.append (self.resnet18)
        self.layers.append (
                nn.LeakyReLU(0.2, inplace=True)
                )
        self.layers.append(
                nn.Linear(1000, Path_length, bias=False),          
                 )
                #self.layers.append (
                #nn.BatchNorm2d(Path_length),
                #   )
        self.layers.append(
                nn.Sigmoid()
                 )
            #self.layers.append(this_layer)
    def forward(self, x):
        #output = self.main(input)
        #layer_len = len(kernels)
        #for layer_point in range(layer_len):
        #    if(layer_len==0):
        #        output = self.layers[layer_point](input)
        #    else:
        #        output = self.layers[layer_point](output)
        #for i, name in enumerate(self.layers):
        #    x = self.layers[i](x)
        for i, name in enumerate(self.layers):
            x = self.layers[i](x)
        return x 

    # mainly based on the resnet     
class _netD_Resnet(nn.Module):
    def __init__(self):
        super(_netD_Resnet, self).__init__()

        #layer_len = len(kernels)
        #create the layer list
        self.layers = nn.ModuleList()
        #self.resnet18 = torchvision.models.resnet18(pretrained = False, **kwargs)
        resnet18 = torchvision.models.resnet18(pretrained = False)

        resnet18.fc = nn.Linear(512, Path_length,bias=False)
        #self.resnet34 = torchvision.models.resnet34(pretrained = False)
        #self.resnet101 = torchvision.models.resnet101(pretrained = False)

        this_input_depth = 2
        this_output_depth  = 3
        #self.layers.append (
        #        nn.Conv2d(this_input_depth, this_output_depth, 3, 1, 0, bias=False),
        #        )
        #self.layers.append (
        #        nn.BatchNorm2d(this_output_depth),
        #           ) 
        #self.layers.append (
        #        nn.ReLU(True)
        #        )
                #self.layers.append (
                #nn.LeakyReLU(0.2, inplace=True)
                #)
        self.layers.append (resnet18)
        #self.layers.append (
        #        nn.LeakyReLU(0.2, inplace=True)
        #        )
        #self.layers.append(
        #        nn.Linear(1000, Path_length, bias=False),          
        #         )
        self.layers.append(
                nn.Sigmoid()
                 )
   
        #self.layers.append(
        #        nn.Tanh()
        #         )
            #self.layers.append(this_layer)
    def forward(self, x):
        #output = self.main(input)
        #layer_len = len(kernels)
        #for layer_point in range(layer_len):
        #    if(layer_len==0):
        #        output = self.layers[layer_point](input)
        #    else:
        #        output = self.layers[layer_point](output)
        #for i, name in enumerate(self.layers):
        #    x = self.layers[i](x)
        for i, name in enumerate(self.layers):
            x = self.layers[i](x)
        return x 

     # mainly based on the resnet     
class _Shifting_Net_(nn.Module):
    def __init__(self):
        super(_Shifting_Net_, self).__init__()

        #layer_len = len(kernels)
        #create the layer list
        self.layers = nn.ModuleList()
        #self.resnet18 = torchvision.models.resnet18(pretrained = False, **kwargs)
        resnet18 = torchvision.models.resnet18(pretrained = True)

        resnet18.fc = nn.Linear(512, 1,bias=False)
        #self.resnet34 = torchvision.models.resnet34(pretrained = False)
        #self.resnet101 = torchvision.models.resnet101(pretrained = False)

        this_input_depth = 2
        this_output_depth  = 3
        #self.layers.append (
        #        nn.Conv2d(this_input_depth, this_output_depth, 3, 1, 0, bias=False),
        #        )
        #self.layers.append (
        #        nn.BatchNorm2d(this_output_depth),
        #           ) 
        #self.layers.append (
        #        nn.ReLU(True)
        #        )
                #self.layers.append (
                #nn.LeakyReLU(0.2, inplace=True)
                #)
        self.layers.append (resnet18)
        #self.layers.append (
        #        nn.LeakyReLU(0.2, inplace=True)
        #        )
        #self.layers.append(
        #        nn.Linear(1000, Path_length, bias=False),          
        #         )
        self.layers.append(
                nn.Sigmoid()
                 )
   
        #self.layers.append(
        #        nn.Tanh()
        #         )
            #self.layers.append(this_layer)
    def forward(self, x):
        #output = self.main(input)
        #layer_len = len(kernels)
        #for layer_point in range(layer_len):
        #    if(layer_len==0):
        #        output = self.layers[layer_point](input)
        #    else:
        #        output = self.layers[layer_point](output)
        #for i, name in enumerate(self.layers):
        #    x = self.layers[i](x)
        for i, name in enumerate(self.layers):
            x = self.layers[i](x)
        return x 
class _Shift_Net_6_layers(nn.Module):
    def __init__(self):
        super(_Shift_Net_6_layers, self).__init__()
        #output width=((W-F+2*P )/S)+1

        kernels = [6, 4, 4, 4, 2,4]
        strides = [2, 2, 2, 2, 2,1]
        pads =    [2, 1, 1, 1, 0,0]
        layer_len = len(kernels)
        #create the layer list
        self.layers = nn.ModuleList()
        for layer_pointer in range(layer_len):
             # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
             # input is (nc) x 128 x 128
            if  layer_pointer ==0:
                this_input_depth = 2
                this_output_depth = 16
            else:
                this_input_depth = this_output_depth
                this_output_depth = this_output_depth*2


    
            if (layer_pointer == (layer_len-1)):
                #self.layers = nn.Sequential(
                #nn.Conv2d(this_input_depth, Path_length, kernels[layer_len -layer_pointer-1], strides[layer_len -layer_pointer-1], pads[layer_len -layer_pointer-1], bias=False),
                #nn.Sigmoid()
                # )
                self.layers.append(
                nn.Conv2d(this_input_depth, 1000, kernels[layer_pointer], strides[layer_pointer], pads[layer_pointer], bias=False),          
                 )
                #self.layers.append (
                #nn.BatchNorm2d(1000),
                #   )
                #self.layers.append(
                #nn. AdaptiveAvgPool2d(output_size=(1, 1)),    
                # )
                self.layers.append (
                nn.LeakyReLU(0.2, inplace=False) #1
                )
                self.layers.append(
                nn.Linear(1000, 1, bias=False),   #2       
                 )
 
                self.layers.append(
                nn.Sigmoid()
                 )
            else:
                  # input is (nc) x 64 x 64
                self.layers.append (
                nn.Conv2d(this_input_depth, this_output_depth, kernels[layer_pointer], strides[layer_pointer], pads[layer_pointer], bias=False),
                )
                self.layers.append (
                nn.BatchNorm2d(this_output_depth),
                   )
                self.layers.append (
                nn.LeakyReLU(0.2, inplace=False)
                )
            #self.layers.append(this_layer)
    def forward(self, x):
        for i, name in enumerate(self.layers):
            x = self.layers[i](x)
            if i == 16 :
                x = x.view(Batch_size,-1).squeeze(1)# squess before fully connected 


        return x 


 

