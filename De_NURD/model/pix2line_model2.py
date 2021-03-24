import torch
from model.base_model import BaseModel
import model.networks as  networks
from test_model import layer_body_sheath_res2
from test_model import fusion_nets3
from test_model.loss_MTL import MTL_loss
#import rendering
Resample_size =256
Batch_size = 1
Path_length = 256
#from dataset_pair_path import  Batch_size,Resample_size, Path_length
from time import time
import torch.nn as nn
from torch.autograd import Variable


class Pix2LineModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        #LGQ
        #parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            #LGQ
            #parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)

        # LGQ here I change the generator to my line encoding
        #self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                              not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG  = fusion_nets3._2layerFusionNets_()
        

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # LGQ add another loss for G
            self.criterionMTL= MTL_loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_f = torch.optim.Adam(self.netG.fusion_layer.  parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_G_1 = torch.optim.Adam(self.netG.side_branch1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_2 = torch.optim.Adam(self.netG.side_branch2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_3 = torch.optim.Adam(self.netG.side_branch3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))




            
            self.optimizers.append(self.optimizer_G)
            

        self.validation_init()
        self.bw_cnt =0
        self.displayloss1=0
        self.displayloss2=0
        self.displayloss3=0

    def validation_init(self):
        self.L1 = 0
        self.L2 = 0
        self.J1 = 0
        self.J2 = 0
        self.J3 = 0

        self.D1 = 0
        self.D2 = 0
        self.D3 = 0
        self.validation_cnt =0
    

    def set_input(self,  pathes,inputG):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """


        # LGQ modify it as one way 
        #AtoB = self.opt.direction == 'AtoB'
        #self.real_A = input['A' if AtoB else 'B'].to(self.device)
        #self.real_B = input['B' if AtoB else 'A'].to(self.device)
        #self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # LGQ modify it as one way 
        #self.real_A = realA.to(self.device)

        #self.real_B = realB.to(self.device)
        #self.real_B=rendering.layers_visualized_integer_encodeing(pathes,Resample_size)
        # LGQ add real path as creterioa for G
        self.real_pathes = pathes 
        self.input_G  = inputG 
    def set_G_input(self,input_G):
        self.input_G  = input_G 
 


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        start_time = time()
        #self.out_pathes = self.netG(self.input_G) # coordinates encoding
        f1,self.out_pathes1,self.path_long1 = self.netG.side_branch1 (self.input_G) # coordinates encoding
        
        f2,self.out_pathes2,self.path_long2 = self.netG.side_branch2 (self.input_G) # coordinates encoding
        f3,self.out_pathes3,self.path_long3 = self.netG.side_branch3 (self.input_G) # coordinates encoding
        self.out_pathes0 =self.netG. fuse_forward( f1,f2,f3)
        test_time_point = time()
        print (" all test point time is [%f] " % ( test_time_point - start_time))
        self.out_pathes = [self.out_pathes0,self.out_pathes1,self.out_pathes2,self.out_pathes3]




        #self.fake_B=  rendering.layers_visualized_integer_encodeing (self.out_pathes3,Resample_size) 
        #self.fake_B_1_hot = rendering.layers_visualized_OneHot_encodeing  (self.out_pathes3,Resample_size) 
        #self.fake_B = self.netG(self.real_A)  # G(A)

     

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        self.optimizer_G.zero_grad()        # set G's gradients to zero

        #self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netG, True)  # D requires no gradients when optimizing G
        # just remain the upsample fusion parameter to optimization 
        #self.set_requires_grad(self.netG.side_branch1, False)  # D requires no gradients when optimizing G
        #self.set_requires_grad(self.netG.side_branch2, False)  # D requires no gradients when optimizing G
        #self.set_requires_grad(self.netG.side_branch3, False)  # D requires no gradients when optimizing G
        #self.set_requires_grad(self.netG.side_branch1.fullout, True)  # D requires no gradients when optimizing G
        #self.set_requires_grad(self.netG.side_branch2.fullout, True)  # D requires no gradients when optimizing G
        #self.set_requires_grad(self.netG.side_branch3.fullout, True)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netG.fusion_layer , True)  # D requires no gradients when optimizing G



        #fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        #pred_fake = self.netD(fake_AB)
        #self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        #self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        #LGQ special fusion loss
        self.loss=self.criterionMTL.multi_loss([self.out_pathes0],self.real_pathes)

        #self.loss_G_L1 =( 1.0*loss[0]  + 0.5*loss[1] + 0.1*loss[2] + 0.2*loss[3])*self.opt.lambda_L1
        #self.loss_G_L1 =( 1.0*loss[0]  + 0.02*loss[1] + 0.02*loss[2]+ 0.02*loss[3]+ 0.02*loss[4]+ 0.02*loss[5])*self.opt.lambda_L1
        #self.loss_G_L1_2 = 0.5*loss[0] 
        #self.loss_G_L1 =( 1.0*loss[0]  +   0.01*loss[1] + 0.01*loss[2] +0.01*loss[3]  )*self.opt.lambda_L1
        self.loss_G_L0 =( self.loss[0]    ) 
         
        self.loss_G = self.loss_G_L0
        #self.loss_G =   self.loss_G_L1

        self.loss_G.backward(retain_graph=True)
        #self.optimizer_G.step()             # udpate G's weights
        self.optimizer_G.step()             # udpate G's weights

    def backward_G_1(self):
        self.optimizer_G.zero_grad()        # set G's gradients to zero
 
        # First, G(A) should fake the discriminator
        #self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netG, True)  # D requires no gradients when optimizing G

        #self.set_requires_grad(self.netG.side_branch1, True)  # D requires no gradients when optimizing G
        #self.set_requires_grad(self.netG.side_branch2, False)  # D requires no gradients when optimizing G
        #self.set_requires_grad(self.netG.side_branch3, False)  # D requires no gradients when optimizing G
        
        loss1 = self.criterionMTL.multi_loss([self.out_pathes1],self.real_pathes) 
        #self.loss_G_L1 = Variable( loss1[0],requires_grad=True)
        self.loss_G_L1 =   loss1[0] 

          
        self.loss_G_L1.backward(retain_graph=True)
        self.optimizer_G_1.step()             # udpate G's weights
    def backward_G_2(self):
        self.optimizer_G.zero_grad()        # set G's gradients to zero
 
        # First, G(A) should fake the discriminator
        #self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netG, True)  # D requires no gradients when optimizing G

        #self.set_requires_grad(self.netG.side_branch1, False)  # D requires no gradients when optimizing G
        #self.set_requires_grad(self.netG.side_branch2, True)  # D requires no gradients when optimizing G
        #self.set_requires_grad(self.netG.side_branch3, False)  # D requires no gradients when optimizing G
         
        loss2 = self.criterionMTL.multi_loss([self.out_pathes2],self.real_pathes) 
        
        self.loss_G_L2 =   loss2[0] 

          
        self.loss_G_L2.backward(retain_graph=True)
        self.optimizer_G_2.step()             # udpate G's weights
    def backward_G_3(self):

        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        self.optimizer_G.zero_grad()        # set G's gradients to zero

        #self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netG, True)  # D requires no gradients when optimizing G

        #self.set_requires_grad(self.netG.side_branch1, False)  # D requires no gradients when optimizing G
        #self.set_requires_grad(self.netG.side_branch2, False)  # D requires no gradients when optimizing G
        #self.set_requires_grad(self.netG.side_branch3, True)  # D requires no gradients when optimizing G

       
         
        loss3 = self.criterionMTL.multi_loss([self.out_pathes3],self.real_pathes) 
        
        self.loss_G_L3 =  loss3[0] 
        #self.loss_G_L3.backward(retain_graph=True)
        self.loss_G_L3.backward( retain_graph=True)

        self.optimizer_G_3.step()             # udpate G's weights

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A) # seperatee the for
        ## update D
        #self.set_requires_grad(self.netD, True)  # enable backprop for D
        #self.optimizer_D.zero_grad()     # set D's gradients to zero
        #self.backward_D()                # calculate gradients for D
        #self.optimizer_D.step()          # update D's weights
        # update G


        self.backward_G_1()                   # calculate graidents for G

        self.backward_G_2()                   # calculate graidents for G

        self.backward_G_3()                   # calculate graidents for G
       
        self.backward_G()                   # calculate graidents for G
        
        self.displayloss0 = self.loss_G_L0. data.mean()

        self.displayloss1 = self.loss_G_L1. data.mean()
        self.displayloss2 = self.loss_G_L2. data.mean()
        self.displayloss3 = self.loss_G_L3. data.mean()

        #if self.  bw_cnt %2 ==0:
        #   self.backward_G()                   # calculate graidents for G
        #else:  
        #    self.backward_G_1()                   # calculate graidents for G
        #    self.backward_G_2()                   # calculate graidents for G
        #    self.backward_G_3()                   # calculate graidents for G
        #    self.displayloss = self.loss_G_L2. data.mean()
        #self.   bw_cnt +=1               # calculate graidents for G
        #if self.   bw_cnt >100:
        #    self.   bw_cnt =0




