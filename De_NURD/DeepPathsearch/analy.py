#save_dir_analys =  "..\\saved_stastics\\"
save_dir_analys =  "..\\dataset\\saved_stastics\\2\\" 

#used python packages
#import keyboard
import cv2
import math
import numpy as np
import os
import random
from matplotlib.pyplot import *
#from mpl_toolkits.mplot3d import Axes3D


#PythonETpackage for xml file edition
try: 
    import xml.etree.cElementTree as ET 
except ImportError: 
    import xml.etree.ElementTree as ET 
import sys 
#GPU acceleration

from analy_visdom import VisdomLinePlotter
from numba import vectorize
from numba import jit
import pickle
from enum import Enum
Save_signal_flag = True
Visidom_flag = False



all_statics_dir = os.path.join(save_dir_analys, 'signals.pkl')
labels = ('mean path error','path cost','additional kp','additional ki')

if (Visidom_flag ==True):
    plotter = VisdomLinePlotter(env_name='Tutorial Plots')

    # this num is speccial for One -D signal
class Save_signal_enum(Enum):
    image_iD =0
    mean_path_error=1
    path_cost  = 2
    additional_kp =3
    additional_ki = 4
     

class MY_ANALYSIS(object):

    def __init__(self ):
         
        #self.signal_label=displayed_labels
        self.DIM= len(Save_signal_enum)
        self.signals= np.zeros((self.DIM,1))
        self.all_statics_dir = os.path.join(save_dir_analys, 'signals.pkl')
        self.path_saving =[]   
        self.first_data_flag = True
        self.path_integral =[]
   # add new step of all signals

    def add_new_iteration_result(self,new_step,this_path):
        #self.signals  = np.append(self.signals,new_step,axis=1) 
        
        if(self.first_data_flag  == True):
            self.first_data_flag = False
            self.path_saving =  this_path
            self.signals  = new_step
        
            #self.path_saving .reshape(len(this_path),1)
        else:
            self.path_saving = np.vstack((self.path_saving , this_path)  )
            self.signals = np.hstack((self.signals , new_step)  )

            #self.path_saving.append(this_path)

            # display and save 2 :using the visdom
    def buffer_path_integral(self, path_intr):
        self.path_integral.append(path_intr)
    def display_and_save2(self,iteration_num,new):
        #save
        if(int(iteration_num)%2==0):
            with open(self.all_statics_dir, 'wb') as f:
                pickle.dump(self , f, pickle.HIGHEST_PROTOCOL)
        if(Visidom_flag==True):
            #plot with visidom
            #every signal
            for i in range(self.DIM):                  
                plotter.plot( Save_signal_enum(i).name, Save_signal_enum(i).name, Save_signal_enum(i).name, iteration_num, new[i,0])
        else:
            #plot method 2
            if(int(iteration_num)%2==0):
                #every signal
                for i in range(self.DIM):
                   figure(i+1)
                   plot( self.signals[i,:], label= Save_signal_enum(i) )
                   xlabel('steps')
                   title(Save_signal_enum(i))   
                draw()
                pause(1e-17) 
                legend()
                   #legend()
    #read from file
    # display and save
    def display_and_save(self,iteration_num):
        #save
        if(int(iteration_num)%2==0):
            with open(self.all_statics_dir, 'wb') as f:
                pickle.dump(self , f, pickle.HIGHEST_PROTOCOL)
        #plot method 2
        if(int(iteration_num)%2==0):
            #every signal
            for i in range(self.DIM):
               figure(i+1)
               plot( self.signals[i,:], label= Save_signal_enum(i) )
               xlabel('steps')
               title(Save_signal_enum(i))   
            draw()
            pause(1e-17)
            time.sleep(0.1)
               #legend()

    #display only
    def display(self):
        for i in range(self.DIM):
               figure(i+1)
               plot( self.signals[i,:], label= Save_signal_enum(i) )
               xlabel('steps')
               title(Save_signal_enum(i))   
        draw()
        pause(1e-17)
        time.sleep(0.1)
               #legend()
    #read from file
    def read_my_signal_results(self):
        self = pickle.load(open(self.all_statics_dir,'rb'),encoding='iso-8859-1')
        return self
 
    #test
    def test_this_class_function():
        
        #test start 
        signals = MY_ANALYSIS()
        signals2 = MY_ANALYSIS()
        kp =1
        ki=2
        kd =3
        for i in range(100):
            kp+=1
            ki+=1
            new = np.zeros((signals.DIM,1))
    
            new[Save_signal_enum.path_cost.value]= kp
            new[Save_signal_enum.mean_path_error.value]= ki
     
            signals.add_new_iteration_result(new)
            signals.display_and_save2(i,new)
        signals2=signals2.read_my_signal_results()
        signals2=signals2.read_my_signal_results()

        #finish test






 



