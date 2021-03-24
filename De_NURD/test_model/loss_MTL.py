import torch.nn as nn
import torch.nn.functional as F
class MTL_loss(object):
   def __init__(self):
       self.criterion = nn.L1Loss()
   def one_loss(self,output1,target1):
       b, _,len = output1.size()
       target_scaled = F.interpolate(target1, size=len, mode='area')
       this_loss = self.criterion(output1, target_scaled)
       return this_loss
   def multi_loss(self,output,target):
       num = len(output)
       loss = [None]*num
       for i in  range(num):
           loss[i] = self.one_loss(output[i],target)
       return loss  
        
