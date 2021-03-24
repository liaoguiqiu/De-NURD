# the NET dbody for the sheath contour tain  upadat 5th octo 2020
import torch
import torch.nn as nn
#import  arg_parse
#from arg_parse import kernels, strides, pads
#from  dataset_pair_path import Path_length,Batch_size,Resample_size
Resample_size =256
Batch_size = 1
Path_length = 256
import torchvision.models
#nz = int( arg_parse.opt.nz)
#ngf = int( arg_parse.opt.ngf)
#ndf = int( arg_parse.opt.ndf)
nc = 1
basic_feature=12

class conv_keep_W(nn.Module):
    def __init__ (self, indepth,outdepth,k=(4,5),s=(2,1),p=(1,2)):
        super(conv_keep_W, self).__init__()
        self.conv_block = self.build_conv_block(indepth,outdepth,k,s,p)
    def build_conv_block(self, indepth,outdepth,k,s,p):
        module = nn.Sequential(
    # relection padding padding_left , \text{padding\_right}padding_right , \text{padding\_top}padding_top , \text{padding\_bottom}padding_bottom 
            #nn.ReflectionPad2d((p[1],p[1],p[0],p[0])), 
            #nn.Conv2d(indepth, outdepth,k, s, (0,0), bias=False),  
            nn.Conv2d(indepth, outdepth,k, s, p, bias=False),          
            
            nn.BatchNorm2d(outdepth),
            #nn.GroupNorm(8*int(outdepth/basic_feature),outdepth),

            nn.LeakyReLU(0.1,inplace=False) # after I add Iddentity afre this the inplace should be false, 
            )                                # next time I should  use identity before the relu layer

        return module
    def forward(self, x):
        #"""Forward function (with skip connections)"""
        out =  self.conv_block(x)  # add skip connections

        # this is a self desined residual block for Deeper nets

        #local_bz,channel,H,W = out.size() 
        #downsample = nn.AdaptiveAvgPool2d((H,W))(x)
        #_,channel2,_,_ = downsample.size() 
        #out[:,0:channel2,:,:] = out[:,0:channel2,:,:]+  downsample
        return out
#output width=((W-F+2*P )/S)+1
# Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
     

class conv_dv_2(nn.Module):
    def __init__ (self, indepth,outdepth,k=(6,6),s=(2,2),p=(2,2)):
        super(conv_dv_2, self).__init__()
        self.conv_block = self.build_conv_block(indepth,outdepth,k,s,p)

    def build_conv_block(self, indepth,outdepth,k,s,p):
        module = nn.Sequential(
             #nn.ReflectionPad2d((p[1],p[1],p[0],p[0])),             
             #nn.Conv2d(indepth, outdepth,k, s,(0,0), bias=False), 
             nn.Conv2d(indepth, outdepth,k, s,p, bias=False),          
             
             nn.BatchNorm2d(outdepth),
             #nn.GroupNorm(8*int(outdepth/basic_feature),outdepth),

             nn.LeakyReLU(0.1,inplace=False)
             )
        return module
 

    def forward(self, x):
        #"""Forward function (with skip connections)"""

        out =  self.conv_block(x)  # add skip connections
        #local_bz,channel,H,W = out.size() 
        #downsample = nn.AdaptiveAvgPool2d((H,W))(x)
        #_,channel2,_,_ = downsample.size() 
        #out[:,0:channel2,:,:] = out[:,0:channel2,:,:]+  downsample
        return out
#output width=((W-F+2*P )/S)+1
# Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    
class conv_keep_all(nn.Module):
    def __init__ (self, indepth,outdepth,k=(3,3),s=(1,1),p=(1,1)):
        super(conv_keep_all, self).__init__()
        self.conv_block = self.build_conv_block(indepth,outdepth,k,s,p)

    def build_conv_block(self, indepth,outdepth,k,s,p):
        module = nn.Sequential(
             #nn.ReflectionPad2d((p[1],p[1],p[0],p[0])), 
             #nn.Conv2d(indepth, outdepth,k, s, (0,0), bias=False),          
             nn.Conv2d(indepth, outdepth,k, s,p, bias=False),          

             nn.BatchNorm2d(outdepth),
             #nn.GroupNorm(4*int(outdepth/basic_feature),outdepth),

             nn.LeakyReLU(0.1,inplace=True)
             )
        return module
 
    def forward(self, x):
        #"""Forward function (with skip connections)"""
        #out = x+ self.conv_block(x)  # add skip connections
        out =  self.conv_block(x)  # add skip connections

        return out

 
# mainly based on the resnet  
class _netD_8_multiscal_fusion300_layer(nn.Module):
#output width=((W-F+2*P )/S)+1

    def __init__(self):
        super(_netD_8_multiscal_fusion300_layer, self).__init__()
        ## depth rescaler: -1~1 -> min_deph~max_deph
        mind= -1
        maxd= 2
        feature = basic_feature

        self.scaler = lambda d : (1+d)/2*maxd + (1-d)/2*mind
        self.layer_num =2
        #a side branch predict with original iamge with rectangular kernel
        # 300*300 - 150*300
        #limit=1024
        self.side_branch1  =  nn.ModuleList()    
        self.side_branch1.append(  conv_keep_W(3,feature))
        # 150*300 - 75*300
        #self.side_branch1.append(  conv_keep_all(feature, feature))
        self.side_branch1.append(  conv_keep_all(feature, feature))

        self.side_branch1.append(  conv_keep_W(feature,2*feature))
        feature = feature *2
        # 75*300  - 37*300
        #self.side_branch1.append(  conv_keep_all(feature, feature))

        self.side_branch1.append(  conv_keep_all(feature, feature))

        self.side_branch1.append(  conv_keep_W(feature,2*feature))
        feature = feature *2
        # 37*300  - 18*300
        self.side_branch1.append(  conv_keep_all(feature, feature))
        self.side_branch1.append(  conv_keep_W(feature,2*feature))
        feature = feature *2
        # 18*300  - 9*300
        #self.side_branch1.append(  conv_keep_all(feature, feature))
        #self.side_branch1.append(  conv_keep_all(feature, feature))
        self.side_branch1.append(  conv_keep_all(feature, feature))

        self.side_branch1.append(  conv_keep_W(feature,2*feature))
        feature = feature *2
        # 9*300  - 4*300
        self.side_branch1.append(  conv_keep_all(feature, feature))
        self.side_branch1.append(  conv_keep_W(feature,2*feature))
        feature = feature *2
        self.side_branch1.append(  conv_keep_all(feature, feature))

        self.side_branch1.append(  conv_keep_W(feature,feature,k=(4,1),s=(1,1),p=(0,0)))
         
        #feature = feature *2
        self.side_branch1.append( nn.Sequential(
              
             nn.Conv2d(feature, self.layer_num,(1,1), (1,1), (0,0), bias=False)         
             #nn.BatchNorm2d(1),
             #nn.LeakyReLU(0.1,inplace=True)
                                                    )
                                 )

        feature = basic_feature

        #a side branch predict with original iamge with rectangular kernel
        # 300*300 - 150*300
        self.side_branch2  =  nn.ModuleList()    
        self.side_branch2.append(  conv_keep_W(3,feature))
        # 150*300 - 75*150

        self.side_branch2.append(  conv_dv_2(feature,2*feature))#
        # 75*150  - 37*150
        feature = feature *2
        #self.side_branch2.append(  conv_keep_all(feature, feature))
        #self.side_branch2.append(  conv_keep_all(feature, feature))

        self.side_branch2.append(  conv_keep_all(feature, feature))

        self.side_branch2.append(  conv_keep_W(feature,2*feature))
        # 37*150  - 18*75
        feature = feature *2
        #self.side_branch2.append(  conv_keep_all(feature, feature))
        #self.side_branch2.append(  conv_keep_all(feature, feature))

        self.side_branch2.append(  conv_keep_all(feature, feature))

        self.side_branch2.append(  conv_dv_2(feature,2*feature))
        # 18*75  - 9*75
        feature = feature *2
        self.side_branch2.append(  conv_keep_all(feature, feature))
        self.side_branch2.append(  conv_keep_W(feature,2*feature))
        # 9*75  - 4*75
        feature = feature *2
        #self.side_branch2.append(  conv_keep_all(feature, feature))
        #self.side_branch2.append(  conv_keep_all(feature, feature))

        self.side_branch2.append(  conv_keep_all(feature, feature))

        self.side_branch2.append(  conv_keep_W(feature,2*feature))

        # 4*75  - 1*75
        feature = feature *2
        self.side_branch2.append(  conv_keep_all(feature, feature))

        self.side_branch2.append(  conv_keep_W(feature,feature,k=(4,1),s=(1,1),p=(0,0)))
        # 1*75  - 1*300
         
        #feature = feature *2
        # use a transpose instead of interpolation to restore 
        #self.side_branch2.append( nn.Sequential(
              
        #     nn.Conv2d(feature, self.layer_num,(1,1), (1,1), (0,0), bias=False)         
        #     #nn.BatchNorm2d(1),
        #     #nn.LeakyReLU(0.1,inplace=True)
        #                                            )
        #                         )
        #has two different out put : one is 75 , another is 300
        self. branch2_out300 = nn.Sequential(
              
             nn.ConvTranspose2d(feature, self.layer_num,(1,4), (1,4), (0,0), bias=False)         
             #nn.BatchNorm2d(1),
             #nn.LeakyReLU(0.1,inplace=True)
                                                    )
        self. branch2_out75 = nn.Sequential(
              
             nn.ConvTranspose2d(feature, self.layer_num,(1,1), (1,1), (0,0), bias=False)       
             #nn.BatchNorm2d(1),
             #nn.LeakyReLU(0.1,inplace=True)
                                                    )
                                 
        self.branch1LU = nn.LeakyReLU(0.1,inplace=True)
        self.branch2LU = nn.LeakyReLU(0.1,inplace=True)
        self.fusion_layer = nn.Conv2d(2*self.layer_num,self.layer_num,(1,3), (1,1), (0,1), bias=False)       
        self.tan_activation = nn.Tanh()
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
             
        

        side_out2 =x
        for j, name in enumerate(self.side_branch2):
            side_out2 = self.side_branch2[j](side_out2)
             
        #side_out2 = nn.functional.interpolate(side_out2, size=(1, Path_length), mode='bilinear') 
        #fusion
        #fuse1=self.branch1LU(side_out)
        side_out_2_300 = self.branch2_out300(side_out2)
        side_out_2_75 = self.branch2_out75 (side_out2)



        fuse1=side_out
        #side_out2 = nn.functional.interpolate(side_out2, size=(1, Path_length), mode='bilinear') 

        #fuse2=self.branch2LU(side_out2)
        fuse2=side_out_2_300

        fuse=torch.cat((fuse1,fuse2),1)
        fuse=self.fusion_layer(fuse)
        #local_bz,_,_,local_l = fuse.size() 

        side_out = side_out.view(-1,self.layer_num,Path_length).squeeze(1)# squess before fully connected
        #side_out = self.tan_activation(side_out)  #  nn.Tanh (side_out)  nn.Tanh()(input)
        #side_out  = self.scaler(side_out)
        side_out_2_300 = side_out_2_300.view(-1,self.layer_num,Path_length).squeeze(1)# squess before fully connected
        #side_out_2_300 = self.tan_activation (side_out_2_300)
        #side_out_2_300  = self.scaler(side_out_2_300)
        local_bz,num,_,local_l = side_out_2_75.size() 
        side_out_2_75 = side_out_2_75.view(-1,num,local_l).squeeze(1)# squess before fully connected
        #side_out_2_75 = self.tan_activation (side_out_2_75)
        #side_out_2_75  = self.scaler(side_out_2_75)



        out  = fuse.view(-1,self.layer_num,Path_length).squeeze(1)# squess before fully connected
        #out = self.tan_activation (out)
        #out  = self.scaler(out)
        out = 0.5 * out + 0.1* side_out_2_300 + 0.4 *side_out
         
        #out
        #return out,side_out,side_out_2_300,side_out_2_75
        return out,side_out_2_75
        
        #return out,side_out,side_out2
# mainly based on the resnet  
