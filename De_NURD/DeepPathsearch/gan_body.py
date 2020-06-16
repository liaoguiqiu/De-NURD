import torch
import torch.nn as nn
import DeepPathsearch.arg_parse
from DeepPathsearch.arg_parse import kernels, strides, pads
from DeepPathsearch.dataset import Path_length,Batch_size,Resample_size
import torchvision.models
nz = int(DeepPathsearch.arg_parse.opt.nz)
ngf = int(DeepPathsearch.arg_parse.opt.ngf)
ndf = int(DeepPathsearch.arg_parse.opt.ndf)
nc = 1


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

class _netD_8_change_outactivation(nn.Module):
    def __init__(self):
        super(_netD_8_change_outactivation, self).__init__()
        kernels = [6, 4, 4, 4, 2,2]
        strides = [2, 2, 2, 2, 2,1]
        pads =    [2, 1, 1, 1, 0,0]
        layer_len = len(kernels)
        #create the layer list
        self.layers = nn.ModuleList()
        for layer_pointer in range(layer_len):
             # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
             # input is (nc) x 128 x 128
            if  layer_pointer ==0:
                this_input_depth = 3
                this_output_depth = 16
            else:
                this_input_depth = this_output_depth
                this_output_depth = this_output_depth*3


    
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
                nn.Linear(1000, Path_length, bias=False),   #2       
                 )
                #self.layers.append (
                #nn.BatchNorm2d(Path_length),
                #   )
                #self.layers.append (
                #nn.BatchNorm2d(Path_length),
                #   )
                #self.layers.append(
                #nn.Sigmoid()
                # )
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
            if i == 15 :
                x = x.view(Batch_size,-1).squeeze(1)# squess before fully connected 


        return x 
# mainly based on the resnet     
class _netD_Resnet(nn.Module):
    def __init__(self):
        super(_netD_Resnet, self).__init__()

        #layer_len = len(kernels)
        #create the layer list
        self.layers = nn.ModuleList()
        #self.resnet18 = torchvision.models.resnet18(pretrained = False, **kwargs)
        self.resnet18 = torchvision.models.resnet18(pretrained = False)
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


    
class _netD_8_multiscal_fusion(nn.Module):
    def __init__(self):
        super(_netD_8_multiscal_fusion, self).__init__()
        kernels = [6, 4, 4, 4, 2,2]
        strides = [2, 2, 2, 2, 2,1]
        pads =    [2, 1, 1, 1, 0,0]
        self.fully_connect_len  =1000
        layer_len = len(kernels)

        #a side branch predict with original iamge with rectangular kernel
        # 71*71 - 35*71
        feature = 8
        self.side_branch1  =  nn.ModuleList()
        self.side_branch1.append( nn.Sequential(
             nn.Conv2d(3, feature,(4,3), (2,1), (1,1), bias=False),          
             nn.BatchNorm2d(feature),
             nn.LeakyReLU(0.1,inplace=True)
            
                                                    )
                                 )
        # 35*71 - 17*71

        self.side_branch1.append( nn.Sequential(
             nn.Conv2d(feature, feature*2,(4,3), (2,1), (1,1), bias=False),          
             nn.BatchNorm2d(feature*2),
             nn.LeakyReLU(0.1,inplace=True)
            
                                                    )
                                 )
        feature = feature *2
        # 17*64  - 8*64

        self.side_branch1.append( nn.Sequential(
             nn.Conv2d(feature, feature*2,(4,3), (2,1), (1,1), bias=False),          
             nn.BatchNorm2d(feature*2),
             nn.LeakyReLU(0.1,inplace=True)
            
                                                    )
                                 )
        feature = feature *2
        # 8*64  - 4*64

        self.side_branch1.append( nn.Sequential(
             nn.Conv2d(feature, feature*2,(4,3), (2,1), (1,1), bias=False),          
             nn.BatchNorm2d(feature*2),
             nn.LeakyReLU(0.1,inplace=True)
            
                                                    )
                                 )
        feature = feature *2
        #self.side_branch1.append( nn.Sequential(
        #     nn.Conv2d(256, 512,(64,3), (1,1), (0,1), bias=False),          
        #     nn.BatchNorm2d(512),
        #     nn.LeakyReLU(0.1,inplace=True)
            
        #                                            )
        #                         )
        self.side_branch1.append( nn.Sequential(
             nn.Conv2d(feature, feature*2,(4,1), (1,1), (0,0), bias=False),          
             nn.BatchNorm2d(feature*2),
             nn.LeakyReLU(0.1,inplace=True)
            
                                                    )
                                 )
        feature = feature *2
        self.side_branch1.append( nn.Sequential(
              
             nn.Conv2d(feature, 1,(1,1), (1,1), (0,0), bias=False)         
             #nn.BatchNorm2d(1),
             #nn.LeakyReLU(0.1,inplace=True)
                                                    )
                                 )

        #create the layer list
        self.layers = nn.ModuleList()
        for layer_pointer in range(layer_len):
             # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
             # input is (nc) x 128 x 128
            if  layer_pointer ==0:
                this_input_depth = 3
                this_output_depth = 32
            else:
                this_input_depth = this_output_depth
                this_output_depth = this_output_depth*3


    
            if (layer_pointer == (layer_len-1)):
                #self.layers = nn.Sequential(
                #nn.Conv2d(this_input_depth, Path_length, kernels[layer_len -layer_pointer-1], strides[layer_len -layer_pointer-1], pads[layer_len -layer_pointer-1], bias=False),
                #nn.Sigmoid()
                # )
                self.layers.append(
                nn.Conv2d(this_input_depth, self.fully_connect_len, kernels[layer_pointer], strides[layer_pointer], pads[layer_pointer], bias=False),          
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
                nn.Linear(self.fully_connect_len, Path_length, bias=False),   #2       
                 )
                #self.layers.append (
                #nn.BatchNorm2d(Path_length),
                #   )
                #self.layers.append (
                #nn.BatchNorm2d(Path_length),
                #   )
                #self.layers.append(
                #nn.Sigmoid()
                # )
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
        #output = self.main(input)
        #layer_len = len(kernels)
        #for layer_point in range(layer_len):
        #    if(layer_len==0):
        #        output = self.layers[layer_point](input)
        #    else:
        #        output = self.layers[layer_point](output)
        #for i, name in enumerate(self.layers):
        #    x = self.layers[i](x)
        side_out =x
        for j, name in enumerate(self.side_branch1):
            side_out = self.side_branch1[j](side_out)
             
        side_out = side_out.view(-1,Path_length).squeeze(1)# squess before fully connected 
        for i, name in enumerate(self.layers):
           x = self.layers[i](x)
           if i == 15 :
               x = x.view(-1,self.fully_connect_len).squeeze(1)# squess before fully connected 

        side_out  = 0.6*side_out 
        x=0.4*x
        out  = side_out.add(x)
        # return x
        # return side_out
        return out


class _netD_8_multiscal_fusion_2(nn.Module):
    def __init__(self):
        super(_netD_8_multiscal_fusion_2, self).__init__()
        kernels = [8, 6, 4, 4, 2,2]
        strides = [2, 2, 2, 2, 2,1]
        pads =    [3, 2, 1, 1, 0,0]
        self.fully_connect_len  =1000
        layer_len = len(kernels)

        #a side branch predict with original iamge with rectangular kernel
        # 71*71 - 35*71
        feature = 16
        self.side_branch1  =  nn.ModuleList()
        self.side_branch1.append( nn.Sequential(
             nn.Conv2d(3, feature,(8,7), (2,1), (3,3), bias=False),          
            #  nn.BatchNorm2d(feature),
             nn.LeakyReLU(0.1,inplace=True)
            
                                                    )
                                 )
        # 35*71 - 17*71

        self.side_branch1.append( nn.Sequential(
             nn.Conv2d(feature, feature*2,(6,5), (2,1), (2,2), bias=False),          
            #  nn.BatchNorm2d(feature*2),
             nn.LeakyReLU(0.1,inplace=True)
            
                                                    )
                                 )
        feature = feature *2
        # 17*64  - 8*64

        self.side_branch1.append( nn.Sequential(
             nn.Conv2d(feature, feature*2,(6,5), (2,1), (2,2), bias=False),          
            #  nn.BatchNorm2d(feature*2),
             nn.LeakyReLU(0.1,inplace=True)
            
                                                    )
                                 )
        feature = feature *2
        # 8*64  - 4*64

        self.side_branch1.append( nn.Sequential(
             nn.Conv2d(feature, feature*2,(4,3), (2,1), (1,1), bias=False),          
            #  nn.BatchNorm2d(feature*2),
             nn.LeakyReLU(0.1,inplace=True)
            
                                                    )
                                 )
        feature = feature *2
        #self.side_branch1.append( nn.Sequential(
        #     nn.Conv2d(256, 512,(64,3), (1,1), (0,1), bias=False),          
        #     nn.BatchNorm2d(512),
        #     nn.LeakyReLU(0.1,inplace=True)
            
        #                                            )
        #                         )
        self.side_branch1.append( nn.Sequential(
             nn.Conv2d(feature, feature*2,(4,1), (1,1), (0,0), bias=False),          
            #  nn.BatchNorm2d(feature*2),
             nn.LeakyReLU(0.1,inplace=True)
            
                                                    )
                                 )
        feature = feature *2
        self.side_branch1.append( nn.Sequential(
              
             nn.Conv2d(feature, 1,(1,1), (1,1), (0,0), bias=False)         
             #nn.BatchNorm2d(1),
             #nn.LeakyReLU(0.1,inplace=True)
                                                    )
                                 )
        self.fusion_layer = nn.Conv2d(feature +1,1,(1,4), (1,1), (0,0), bias=False) 

        #create the layer list
        self.layers = nn.ModuleList()
        for layer_pointer in range(layer_len):
             # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
             # input is (nc) x 128 x 128
            if  layer_pointer ==0:
                this_input_depth = 3
                this_output_depth = 32
            else:
                this_input_depth = this_output_depth
                this_output_depth = this_output_depth*3


    
            if (layer_pointer == (layer_len-1)):
                #self.layers = nn.Sequential(
                #nn.Conv2d(this_input_depth, Path_length, kernels[layer_len -layer_pointer-1], strides[layer_len -layer_pointer-1], pads[layer_len -layer_pointer-1], bias=False),
                #nn.Sigmoid()
                # )
                self.layers.append(
                nn.Conv2d(this_input_depth, self.fully_connect_len, kernels[layer_pointer], strides[layer_pointer], pads[layer_pointer], bias=False),          
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
                nn.Linear(self.fully_connect_len, Path_length, bias=False),   #2       
                 )
                #self.layers.append (
                #nn.BatchNorm2d(Path_length),
                #   )
                #self.layers.append (
                #nn.BatchNorm2d(Path_length),
                #   )
                #self.layers.append(
                #nn.Sigmoid()
                # )
            else:
                  # input is (nc) x 64 x 64
                self.layers.append (
                nn.Conv2d(this_input_depth, this_output_depth, kernels[layer_pointer], strides[layer_pointer], pads[layer_pointer], bias=False),
                )
                # self.layers.append (
                # nn.BatchNorm2d(this_output_depth),
                #    )
                self.layers.append (
                nn.LeakyReLU(0.2, inplace=False)
                )
            #self.layers.append(this_layer)
        self.branch1LU = nn.LeakyReLU(0.1,inplace=False)
        self.branch2LU = nn.LeakyReLU(0.1,inplace=False)
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
        side_out =x
        length  = len(self.side_branch1)
        for j, name in enumerate(self.side_branch1):
            # if(isinstance(self.side_branch1[j],nn.BatchNorm2d)):
            #    pass
            # else:
            if j==(length -1):
                side_feature  = side_out
            side_out = self.side_branch1[j](side_out)
             
        for i, key in enumerate(self.layers):
        #    if(isinstance(self.layers[i],nn.BatchNorm2d)):
        #        pass
        #    else:
           x = self.layers[i](x)
           if i == (len(self.layers)-2)  :
               x = x.view(-1,self.fully_connect_len).squeeze(1)# squess before fully connected 
        #fusion
        fuse1=self.branch1LU(side_feature)
        #fuse1=side_feature

        x = x.view(-1,1,Path_length).unsqueeze(1)

        fuse2=self.branch2LU(x)
        #fuse2=x

        fuse=torch.cat((fuse1,fuse2),1)
        fuse=self.fusion_layer(fuse)
        
        side_out = side_out.view(-1,Path_length).squeeze(1)# squess before fully connected
        side_out2 = x.view(-1,Path_length).squeeze(1)# squess before fully connected
        local_bz,_,_,local_l = fuse.size() 
        out = nn.functional.interpolate(fuse, size=(1, Path_length), mode='bilinear') 
        out  = out.view(-1,Path_length).squeeze(1)# squess before fully connected
        #out  = 0.5*out + 0.3 * side_out + 0.2 * side_out2
        # return x
        # return side_out
        return out#,side_out,side_out2