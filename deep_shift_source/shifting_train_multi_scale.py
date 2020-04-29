# rhis is the main py for group shifting train

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import ShiftingNetBody_V2
import arg_parse
import imagenet
from analy import MY_ANALYSIS
from analy import Save_signal_enum
import cv2
import numpy
from image_trans import BaseTransform  
import os
from dataset_shifting import myDataloader_for_shift,Batch_size,Resample_size, Path_length, Mat_size,Resample_size2
from dataset_shifting import Original_window_Len
from multiscaleloss import multiscaleEPE
from  matlab import Save_Signal_matlab

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Switch control for the Visdom or Not
Visdom_flag  = True 

if Visdom_flag == True:
    from analy_visdom import VisdomLinePlotter
    plotter = VisdomLinePlotter(env_name='path finding training Plots')



from scipy import signal 
Matrix_dir =  "..\\dataset\\CostMatrix\\1\\"
#Save_pic_dir = '..\\DeepPathFinding_Version2\\out\\'
opt = arg_parse.opt
opt.cuda = True
# check the cuda device 
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
dataroot = "..\\dataset\\CostMatrix\\"

torch.set_num_threads(2)
######################################################################
# Data
# ----
# 
# In this tutorial we will use the `Celeb-A Faces
# dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`__ which can
# be downloaded at the linked site, or in `Google
# Drive <https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg>`__.
# The dataset will download as a file named *img_align_celeba.zip*. Once
# downloaded, create a directory named *celeba* and extract the zip file
# into that directory. Then, set the *dataroot* input for this notebook to
# the *celeba* directory you just created. The resulting directory
# structure should be:
# 
# ::
# 
#    /path/to/celeba
#        -> img_align_celeba  
#            -> 188242.jpg
#            -> 173822.jpg
#            -> 284702.jpg
#            -> 537394.jpg
#               ...
# 
# This is an important step because we will be using the ImageFolder
# dataset class, which requires there to be subdirectories in the
# dataset’s root folder. Now, we can create the dataset, create the
# dataloader, set the device to run on, and finally visualize some of the
# training data.
# 
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.002)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# We can use an image folder dataset the way we have it setup.
# Create the dataset
 
nz = int(arg_parse.opt.nz) # number of latent variables
ngf = int(arg_parse.opt.ngf) # inside generator
ndf = int(arg_parse.opt.ndf) # inside discriminator
nc = 3 # channels

# custom weights initialization called on netG and netD


#netG = gan_body._netG()
#netG.apply(weights_init)
#if opt.netG != '':
#    netG.load_state_dict(torch.load(opt.netG))
#print(netG)


#netD = gan_body._netD()
#Guiqui 8 layers version
#netD = gan_body._netD_8()

#Guiqiu Resnet version

netD = ShiftingNetBody_V2.ShiftingNet_init( None)
#my_netD  = ShiftingNetBody.ShiftingNet_init_my( None)

netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)


## rename the saved weight
#state_dict = netD.state_dict()
#state_dict_v2 =  netD.state_dict()
#for key in state_dict:
#    if 'flow' in key:
#        pre, post = key.split('flow')
#        state_dict_v2[pre+'shift'+post] = state_dict_v2.pop(key)

#my_netD.load_state_dict(state_dict_v2)
#result =(my_netD.predict_shift6.weight == netD.predict_flow6.weight).all()
#torch.save(my_netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, 0)) 
 


#criterion = nn.BCELoss()
#criterion = nn.MSELoss()
criterion = nn.L1Loss()

#criterion = nn.CrossEntropyLoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    print("CUDA TRUE")
    netD.cuda()
    #netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()


fixed_noise = Variable(fixed_noise)

# setup optimizer
#optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#optimizerD = optim.SGD(netD.parameters(), lr=opt.lr,momentum= 0.9, weight_decay =2e-4 )


#saved_stastics = MY_ANALYSIS()
#saved_stastics=saved_stastics.read_my_signal_results()
#saved_stastics.display()

read_id =0

epoch=0
#transform = BaseTransform(  Resample_size,(104/256.0, 117/256.0, 123/256.0))
#transform = BaseTransform(  Resample_size,[104])  #gray scale data
iteration_num =0
mydata_loader = myDataloader_for_shift (Batch_size,Resample_size,Path_length)
multi_scale_weight = [0.005, 0.01, 0.02, 0.08, 0.32]
matlab_saver  = Save_Signal_matlab()
while(1):
    
    epoch+= 1
    #almost 900 pictures
    while(1):
        iteration_num +=1
        read_id+=1
        if (mydata_loader.read_all_flag ==1):
            read_id =0
            mydata_loader.read_all_flag =0
            break


        mydata_loader.read_a_batch()
        #change to 3 chanels
        #ini_input = mydata_loader.input_mat
        #np_input = numpy.append(ini_input,ini_input,axis=1)
        #np_input = numpy.append(np_input,ini_input,axis=1)
        ini_input_pair1 = mydata_loader.input_pair1
        ini_input_pair2 = mydata_loader.input_pair2
        ini_input_0   = ini_input_pair2
        np_input = numpy.append(mydata_loader.input_pair1,mydata_loader.input_pair2,axis=1)
        np_input = numpy.append(np_input,mydata_loader.input_pair3,axis=1)
        np_input = numpy.append(np_input,mydata_loader.input_pair4,axis=1)


        #np_input = numpy.append(np_input,ini_input_pair2,axis=1)

        input = torch.from_numpy(numpy.float32(np_input)) 
        #input = input.to(device) 
        #input = torch.from_numpy(numpy.float32(mydata_loader.input_image[0,:,:,:])) 
        input = input.to(device)                
   
        patht= torch.from_numpy(numpy.float32(mydata_loader.input_path) )
        #patht=patht.to(device)
                
        #patht= torch.from_numpy(numpy.float32(mydata_loader.input_path[0,:])/71.0 )
        patht=patht.to(device)
        #inputv = Variable(input)
        # using unsqueeze is import  for with out bactch situation
        #inputv = Variable(input.unsqueeze(0))

        #labelv = Variable(patht)
        #inputv = input

        #labelv = patht
        inputv = Variable(input )
        #inputv = Variable(input.unsqueeze(0))
        #patht =patht.view(-1, 1).squeeze(1)

        target = Variable(patht.unsqueeze(2).unsqueeze(1))
        output = netD(inputv)
        loss = multiscaleEPE(output, target, multi_scale_weight, False)

        #output = output.view(Batch_size,Path_length).squeeze(1)
        save_out  = output
        netD.zero_grad()

        #errD_real = criterion(output, labelv)
        #errD_real.backward()
        loss.backward()

        D_x = loss.data.mean()
        optimizerD.step()
        # train with fake
        if cv2.waitKey(12) & 0xFF == ord('q'):
              break 
         
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, 0, read_id, 0,
                 loss.data, D_x, 0, 0, 0))
        if matlab_saver.flag  == True : 
            matlab_saver.buffer_loss(D_x.cpu().detach().numpy())
            matlab_saver.save_mat()
        if Visdom_flag == True:
                plotter.plot( 'LOSS', 'LOSS', 'LOSS', iteration_num, D_x.cpu().detach().numpy())
        if read_id % 2 == 0:
            #vutils.save_image(real_cpu,
            #        '%s/real_samples.png' % opt.outf,
            #        normalize=True)
            #netG.eval()
            #fake = netG(fixed_noise)
            #cv2.imwrite(Save_pic_dir  + str(i) +".jpg", mat)
            #show the result

            dispay_id =0
            Matrix  =   mydata_loader.input_mat[dispay_id,0,:,:] +104
            show1 = Matrix*0
            path2 = (mydata_loader.input_path[dispay_id,:])*Original_window_Len
             
            show1[int(path2[0]),:]=254
             
 
            save_out = save_out[4] 
            save_out = save_out[dispay_id] 


            save_out  = (save_out.data.mean()) *(Original_window_Len)  
            #save_out  = signal.resample(save_out, Mat_size)
            if save_out >= Original_window_Len:
                save_out  = Original_window_Len -1
            show2  = Matrix*0
            
            if save_out >= Original_window_Len:
                save_out  = Original_window_Len-1
            show2[int(save_out ),:]=254


            
            show3 = numpy.append(show1,show2,axis=1) # cascade
            cv2.imshow('Deeplearning one',show3.astype(numpy.uint8)) 
            show4 =  mydata_loader.input_pair1[dispay_id,0,:,:]
            show5 =  mydata_loader.input_pair2[dispay_id,0,:,:]
            show6 =  mydata_loader.input_pair3[dispay_id,0,:,:]
            show7 =  mydata_loader.input_pair4[dispay_id,0,:,:]

            show8 = numpy.append(show4,show5,axis=0)  # cascade
            show8 = numpy.append(show8,show6,axis=0)  # cascade

            show8 = numpy.append(show8,show7,axis=0)  # cascade
            show8  = show8 + 104
            cv2.imshow('Input one',show8.astype(numpy.uint8)) 

            if cv2.waitKey(12) & 0xFF == ord('q'):
              break
    # do checkpointing
    #torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    #torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    #cv2.imwrite(Save_pic_dir  + str(epoch) +".jpg", show2)

    cv2.imwrite(opt.outf  + str(epoch) +".jpg", show2)
    if epoch >=50:
        epoch =0
