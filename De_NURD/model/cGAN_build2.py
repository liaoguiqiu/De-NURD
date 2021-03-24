import time
import torch
from model.options.train_options import TrainOptions
from model.base_model import BaseModel
#from data import create_dataset
#from models import create_model
#from util.visualizer import Visualizer
import importlib
#import model.pix2pix_model as pix2pix_model
import model.pix2line_model2 as pix2line_model
#import model.pix2line_model2 as pix2line_model



class CGAN_creator(object):
    def __init__(self):
        pass
    def creat_cgan(self):
        opt = TrainOptions().parse()   # get training options
    #dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    #dataset_size = len(dataset)    # get the number of images in the dataset.
    #print('The number of training images = %d' % dataset_size)
        
        model = create_model(opt)      # create a model given opt.model and other options
        #get_option_setter
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        #model.opt = model.modify_commandline_options(opt)
        #visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
        
        return model

def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "model." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    #LGQ  direct choose the model 
    #model = find_model_using_name(opt.model)
    #LGQ  direct choose the model 
    #model = pix2pix_model.Pix2PixModel(opt)
    instance = pix2line_model.Pix2LineModel(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance

if __name__ == '__main__':
 
    #dataset_size = len(dataset)    # get the number of images in the dataset.
    #print('The number of training images = %d' % dataset_size)
    realA  = torch.zeros([5,1, 64,64], dtype=torch.float)
    realA=realA.cuda()
    realB  = torch.zeros([5,1, 64,64], dtype=torch.float)
    realB=realB.cuda()
    modeler  = CGAN_creator()
    model = modeler.creat_cgan()
 
    
    model.update_learning_rate()    # update learning rates in the beginning of every epoch.
    model.set_input(realA,realB)         # unpack data from dataset and apply preprocessing
    model.optimize_parameters()   # calculate loss functions, get gradients, update network weights


