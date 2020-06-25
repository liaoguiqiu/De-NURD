import torch
import torch.nn as nn
import torchvision.models
import math
Path_length =832
 

def conv_keep_W(indepth,outdepth,k=(4,3),s=(2,1),p=(1,1)):
#output width=((W-F+2*P )/S)+1
# Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    module = nn.Sequential(
             nn.Conv2d(indepth, outdepth,k, s, p, bias=False),          
             nn.BatchNorm2d(outdepth),
             nn.LeakyReLU(0.1,inplace=True)
             )
    return module
def conv_dv_2(indepth,outdepth,k=(4,4),s=(2,2),p=(1,1)):
#output width=((W-F+2*P )/S)+1
# Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    module = nn.Sequential(
             nn.Conv2d(indepth, outdepth,k, s, p, bias=False),          
             nn.BatchNorm2d(outdepth),
             nn.LeakyReLU(0.1,inplace=True)
             )
    return module
def conv_keep(indepth,outdepth,k=(3,3),s=(1,1),p=(1,1)):
#output width=((W-F+2*P )/S)+1
# Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    module = nn.Sequential(
             nn.Conv2d(indepth, outdepth,k, s, p, bias=False),          
             #nn.BatchNorm2d(outdepth),
             nn.LeakyReLU(0.1,inplace=True)
             )
    return module
    
class _netD_8_multiscal_fusion_long(nn.Module):
    def __init__(self):
        super(_netD_8_multiscal_fusion_long, self).__init__()
        kernels = [6, 6, 4, 4, 2,2]
        strides = [2, 2, 2, 2, 2,1]
        pads =    [2, 2, 1, 1, 0,0]
        self.fully_connect_len = 1000
        layer_len = len(kernels)

        #a side branch predict with original iamge with rectangular kernel
        # 71*832 - 35*832
        feature = 8
        self.side_branch1  =  nn.ModuleList()
        #self.side_branch1.append( conv_keep(3,feature))
        self.side_branch1.append( conv_keep_W(3,2*feature))
        feature = feature *2

        # 35*832 - 17*832
        #self.side_branch1.append( conv_keep(feature,feature))
        self.side_branch1.append( conv_keep_W(feature,2*feature))
        feature = feature *2
        # 17*832  - 8*832
        #self.side_branch1.append( conv_keep(feature,feature))
        self.side_branch1.append( conv_keep_W(feature,2*feature))
        feature = feature *2
        # 8*64  - 4*832
        #self.side_branch1.append( conv_keep(feature,feature))
        self.side_branch1.append( conv_keep_W(feature,2*feature))
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

        # fully connect fuion

        feature = 12
        self.side_branch2  =  nn.ModuleList()
        # 71*832 - 35*416

        #self.side_branch1.append( conv_keep(3,feature))
        self.side_branch2.append( conv_dv_2 (3,2*feature))
        feature = feature *2

        # 35*416 - 17*208
        #self.side_branch1.append( conv_keep(feature,feature))
        self.side_branch2.append( conv_dv_2(feature,2*feature))
        feature = feature *2
        # 17*208  - 8*104
        #self.side_branch1.append( conv_keep(feature,feature))
        self.side_branch2.append( conv_dv_2(feature,2*feature))
        feature = feature *2
        # 8*104  - 4*52
        #self.side_branch1.append( conv_keep(feature,feature))
        self.side_branch2.append( conv_dv_2(feature,2*feature))
        feature = feature *2
        #self.side_branch1.append( nn.Sequential(
        #     nn.Conv2d(256, 512,(64,3), (1,1), (0,1), bias=False),          
        #     nn.BatchNorm2d(512),
        #     nn.LeakyReLU(0.1,inplace=True)
            
        #                                            )
        #                         )
        self.side_branch2.append( nn.Sequential(
             nn.Conv2d(feature, feature*2,(4,1), (1,1), (0,0), bias=False),          
             nn.BatchNorm2d(feature*2),
             nn.LeakyReLU(0.1,inplace=True)
            
                                                    )
                                 )
        feature = feature *2
        self.side_branch2.append( nn.Sequential(
              
             nn.Conv2d(feature, 1,(1,1), (1,1), (0,0), bias=False)         
             #nn.BatchNorm2d(1),
             #nn.LeakyReLU(0.1,inplace=True)
                                                    )
                                 )

            #self.layers.append(this_layer)
        self.branch1LU = nn.LeakyReLU(0.1,inplace=False)
        self.branch2LU = nn.LeakyReLU(0.1,inplace=False)
        self.fusion_layer = nn.Conv2d(2,1,(1,3), (1,1), (0,1), bias=False)       

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
            test=side_out[0,0,0,0].cpu().detach().numpy()
            if math.isnan(test):
                path = 0

                pass
             
        

        side_out2 =x
        for j, name in enumerate(self.side_branch2):
            side_out2 = self.side_branch2[j](side_out2)
             

        #fusion
        fuse1=self.branch1LU(side_out)
        side_out2 = nn.functional.interpolate(side_out2, size=(1, Path_length), mode='bilinear') 

        fuse2=self.branch2LU(side_out2)

        fuse=torch.cat((fuse1,fuse2),1)
        fuse=self.fusion_layer(fuse)
        #local_bz,_,_,local_l = fuse.size() 

        side_out = side_out.view(-1,Path_length).squeeze(1)# squess before fully connected
        side_out2 = side_out2.view(-1,Path_length).squeeze(1)# squess before fully connected

        out  = fuse.view(-1,Path_length).squeeze(1)# squess before fully connected
        out = 0.4*out + 0.3*side_out+ 0.3*side_out2
        #out  = side_out
        # return x
        # return side_out
        return out#,side_out,side_out2
        # return x
       