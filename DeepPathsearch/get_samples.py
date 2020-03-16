import os
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
import gan_body
from arg_parse import opt
import re

# /media/annusha/BigPapa/Study/DL/out

# how to launch this file for rooms
# python3 get_samples.py --niter=1000 --outf=./conference_room/
# --netG=/media/annusha/BigPapa/Study/DL/out/netG_epoch_24.pth --nz=100 --ngf=64

# for imagenet
# --niter=1 --outf=./conference_room/ --dataset=imagenet --imageSize=32
# --dataroot=/media/annusha/BigPapa/Study/DL/out_imagenet
opt.netG  = "..\\DCGANproject\\out\\netG_epoch_1330.pth"
generatepath = "..\\DCGANproject\\generated\\"
def _create_and_save(netG):
     

    for i in range(1, 10000):
        noise = torch.FloatTensor(1, opt.nz, 1, 1).normal_(0, 1)
        noise = Variable(noise)
        noise = noise.cuda()

        fake = netG(noise)
        vutils.save_image(fake.data, generatepath + '%d.jpg' % i, normalize=True )

if __name__ == '__main__':
    netG = gan_body._netG()

 
    print('load weights for generator')
     
    netG.load_state_dict(torch.load(opt.netG))
    print(netG)
    netG.cuda()
    netG.eval()

    _create_and_save(netG)
