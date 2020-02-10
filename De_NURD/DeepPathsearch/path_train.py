import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import gan_body
import arg_parse
import imagenet
from analy import MY_ANALYSIS
from analy import Save_signal_enum
import cv2
import numpy
from image_trans import BaseTransform  
import os
from dataset import myDataloader,Batch_size,Resample_size, Path_length
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



from scipy import signal 
Matrix_dir =  "..\\dataset\\CostMatrix\\1\\"
Save_pic_dir = '..\\DeepPathFinding\\out\\'
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

# We can use an image folder dataset the way we have it setup.
# Create the dataset
 
nz = int(arg_parse.opt.nz) # number of latent variables
ngf = int(arg_parse.opt.ngf) # inside generator
ndf = int(arg_parse.opt.ndf) # inside discriminator
nc = 3 # channels

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


netG = gan_body._netG()
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


#netD = gan_body._netD()
#Guiqui 8 layers version
netD = gan_body._netD_8()

netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    print("CUDA TRUE")
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()


fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


#saved_stastics = MY_ANALYSIS()
#saved_stastics=saved_stastics.read_my_signal_results()
#saved_stastics.display()

read_id =0

epoch=0
#transform = BaseTransform(  Resample_size,(104/256.0, 117/256.0, 123/256.0))
#transform = BaseTransform(  Resample_size,[104])  #gray scale data

mydata_loader = myDataloader (Batch_size,Resample_size,Path_length)
while(1):
    epoch+= 1
    #almost 900 pictures
    while(1):
        read_id+=1
        if (mydata_loader.read_all_flag ==1):
            read_id =0
            mydata_loader.read_all_flag =0
            break


        mydata_loader .read_a_batch()
            
        input = torch.from_numpy(numpy.float32(mydata_loader.input_image)) 
        #input = input.to(device) 
        #input = torch.from_numpy(numpy.float32(mydata_loader.input_image[0,:,:,:])) 
        input = input.to(device)                
   
        patht= torch.from_numpy(numpy.float32(mydata_loader.input_path)/71.0 )
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

        labelv = Variable(patht)
        output = netD(inputv)
        output = output.view(Batch_size,Path_length).squeeze(1)
        save_out  = output
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = errD_real.data.mean()
        optimizerD.step()
        # train with fake
        if cv2.waitKey(12) & 0xFF == ord('q'):
              break 
         
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, 0, read_id, 0,
                 errD_real.data, 0, 0, 0, 0))
        if read_id % 2 == 0:
            #vutils.save_image(real_cpu,
            #        '%s/real_samples.png' % opt.outf,
            #        normalize=True)
            #netG.eval()
            #fake = netG(fixed_noise)
            #cv2.imwrite(Save_pic_dir  + str(i) +".jpg", mat)
            #show the result


            gray2  =   mydata_loader.input_image[0,0,:,:] +104
            show1 = gray2.astype(float)
            path2 = mydata_loader.input_path[0,:]/71*(Resample_size-2)
            path2  = signal.resample(path2, Resample_size)

            for i in range ( len(path2)):
                if path2[i]>=Resample_size:
                   path2[i] = Resample_size-1
                show1[int(path2[i]),i]=254
             
            show2 =  gray2.astype(float)
            save_out = save_out.cpu().detach().numpy()

            save_out  = save_out[0,:] *(Resample_size-2)
            save_out  = signal.resample(save_out, Resample_size)

            for i in range ( len(save_out)):
                if save_out[i]>=Resample_size:
                    save_out[i]=Resample_size-1
                show2[int(save_out[i]),i]=254


            
            show3 = numpy.append(show1,show2,axis=1) # cascade
            cv2.imshow('Deeplearning one',show3.astype(numpy.uint8)) 

            if cv2.waitKey(12) & 0xFF == ord('q'):
              break
    # do checkpointing
    #torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    cv2.imwrite(Save_pic_dir  + str(epoch) +".jpg", show2)
