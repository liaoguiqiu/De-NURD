import torch
import torch.nn as nn
import DeepPathsearch.arg_parse
from DeepPathsearch.arg_parse import kernels, strides, pads
from DeepPathsearch.dataset import Path_length ,Batch_size
import torchvision.models
nz = int(DeepPathsearch.arg_parse.opt.nz)
ngf = int(DeepPathsearch.arg_parse.opt.ngf)
ndf = int(DeepPathsearch.arg_parse.opt.ndf)
nc = 1

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
  
class _Path_net(nn.Module):
    def __init__(self):
        super(_Path_net, self).__init__()

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
                nn.Linear(1000, 64, bias=False),   #2       
                 )
                #self.layers.append (
                #nn.BatchNorm2d(Path_length),
                #   )
                #self.layers.append (
                #nn.BatchNorm2d(Path_length),
                #   )
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
        #output = self.main(input)
        #layer_len = len(kernels)
        #for layer_point in range(layer_len):
        #    if(layer_len==0):
        #        output = self.layers[layer_point](input)
        #    else:
        #        output = self.layers[layer_point](output)


        #  forward version for square image
        #for i, name in enumerate(self.layers):
        #    x = self.layers[i](x)
        #    if i ==15:
        #        x = x.view(Batch_size, -1).squeeze(1)

        #  forward version for a long path   
        for i, name in enumerate(self.layers):
            if i< 15:
              x = self.layers[i](x)
            if i ==15:
                x = self.layers[i](x)
                sub  = x.view(Batch_size, 1000,-1).squeeze(1)
                #x = x.view(Batch_size, -1).squeeze(1)
                #sub_ex  = x[:,:,:,0]
                #sub_ex = sub_ex.view(Batch_size, -1).squeeze(1)
                #_,sub_len   =  shape(sub_ex)
                #sub  = torch.zeros([Batch_size, subsub_len,12], dtype=torch.float32)
                #for j in range(12):
                #    this_sub  = x[:,:,:,j]
                #    sub[:,:,j]  = this_sub.view(Batch_size, -1).squeeze(1)
                    #x1 = x[:,:,:,0]
                    #x2 = x[:,:,:,1]
                    #x3 = x[:,:,:,2]
                    #x1 = x1.view(Batch_size, -1).squeeze(1)
                    #x2 = x2.view(Batch_size, -1).squeeze(1)
                    #x3 = x3.view(Batch_size, -1).squeeze(1)
            if i >=16:

                for  j in range(23):
                    if j ==0 :
                        this_predict  =   self.layers[i](sub[:,:,j])
                        this_predict  =   this_predict.unsqueeze(2)
                    else:
                        new   = self.layers[i](sub[:,:,j])
                        new  = new .unsqueeze(2)
                        this_predict  = torch .cat ((this_predict,new ),dim=2)
                sub  = this_predict
                #x1 = self.layers[i](x1)
                #x2 = self.layers[i](x2)
                #x3 = self.layers[i](x3)
            if i==18:
                small  = torch.zeros([Batch_size, 32,24], dtype=torch.float32)
                small=small.cuda()
                for  j in range(24):
                    if  j ==0 :
                        small[:,:,j]  = sub[:,0:32,j]
                        output  = small[:,:,j]
                        pass
                    if j > 0 and j <23:
                        small[:,:,j]  =( sub[:,32:64,j-1] +  sub[:,0:32,j] )/2
                        output  =  torch .cat ((output,small[:,:,j] ),dim=1)
                        pass
                    if j == 23:
                        small[:,:,j]  = sub[:,32:64,j-1]
                        output  =  torch .cat ((output,small[:,:,j] ),dim=1)
                        pass

                #sub1 = x1[:, 0:32]
                #sub2 = (x1[:, 32:64] + x2[:, 0:32])/2
                #sub3 = (x2[:, 32:64] + x3[:, 0:32])/2
                #sub4 = x3[:, 32:64]

                #x  = torch .cat ((sub1,sub2),dim=1)
                #x  = torch .cat ((x,sub3),dim=1)
                #x  = torch .cat ((x,sub4),dim=1)


        return output


# mainly based on the resnet     
class _netD_Resnet(nn.Module):
    def __init__(self):
        super(_netD_Resnet, self).__init__()
        self.finetune_weight='..\\DeepPathFinding\\out_back\\netD_epoch_1.pth'

        #layer_len = len(kernels)
        #create the layer list
        self.layers = nn.ModuleList()
        #self.resnet18 = torchvision.models.resnet18(pretrained = False, **kwargs)
        self.resnet18 = torchvision.models.resnet18(pretrained = False)
        #self.resnet34 = torchvision.models.resnet34(pretrained = False)
        #self.resnet101 = torchvision.models.resnet101(pretrained = False)
        #self.resnet18.avgpool = nn.LeakyReLU(0.2, inplace=True)
        #self.resnet18.avgpool = nn. AdaptiveAvgPool2d(output_size=(5, 5))
        #self.resnet18.fc = nn.Linear(512 ,1000)
        



        self.layers.append (self.resnet18) #0
        self.layers.append (
                nn.LeakyReLU(0.2, inplace=True) #1
                )
        self.layers.append(
                nn.Linear(1000, 71, bias=False),   #2       
                 )
                #self.layers.append (
                #nn.BatchNorm2d(Path_length),
                #   )
        self.layers.append(
                nn.Sigmoid()
                 )
        # initialization with weight
        self.load_state_dict(torch.load(self.finetune_weight))

        # do the modification to get different output 
        self.layers[0].avgpool = nn.AdaptiveAvgPool2d(output_size=(2, 2)) 
        #self.layers[0].avgpool.apply(weights_init)
        self.layers[0].fc = nn.Linear(512*4, 1000*2, bias=False)  
        #self.layers[0].fc.apply(weights_init)
        # doulbe out put 
        self.layers[2]  =  nn.Linear(1000*2, Path_length, bias=False)   #2      
        #self.layers[2].apply(weights_init)
        
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
