
#used python packages
import cv2
import math
import numpy as np
import os
import random

class Basic_oper(object):


    def tranfer_frome_cir2rec(gray):
        H,W = gray.shape
        value = np.sqrt(((H/2.0)**2.0)+((W/2.0)**2.0))

        polar_image = cv2.linearPolar(gray,(W/2, H/2), value, cv2.WARP_FILL_OUTLIERS)

        polar_image = polar_image.astype(np.uint8)
        polar_image=cv2.rotate(polar_image,rotateCode = 0) 
        return polar_image
    def tranfer_frome_rec2cir(gray):
        H,W = gray.shape
        value = np.sqrt(((H/2.0)**2.0)+((W/2.0)**2.0))
        gray=cv2.rotate(gray,rotateCode = 2) 
        #value = 200
        #circular = cv2.linearPolar(new_frame, (new_frame.shape[1]/2 , new_frame.shape[0]/2), 
        #                               200, cv2.WARP_INVERSE_MAP)
        circular = cv2.linearPolar(gray,(W/2, H/2), value, cv2.WARP_INVERSE_MAP)

        circular = circular.astype(np.uint8)
        #polar_image=cv2.rotate(polar_image,rotateCode = 0) 
        return circular


class Basic_Operator:
    # use the H and W of origina to confine , and generate a random reseanable signal in the window
    def random_shaped_layer (H,W):
        # currently use 4 bondaries
        contoursx=[None]*4
        contoursy=[None]*4
        contoursx[0]   = contoursx[1] = contoursx[2] = contoursx[3] = np.arange(0, W)
        dy1 = 0.1*H
        dy2  = 0.3*H
        for i in range(4):
            # randon contour 
            dy_this1  = dy1  + 0.2*i*H
            dy_this2  = dy2  + 0.2*i*H
            # scale to avoisd the contact beteen 2  bondaties
            dy_this1 = dy_this1*0.8
            dy_this2 = dy_this2*0.8

            r_vector   = np.random.sample(20)*50
            r_vector=signal.resample(r_vector, W)
            r_vector = gaussian_filter1d (r_vector ,10)


            newy =  r_vector
            miny=min(newy)
            height =  dy_this2-dy_this1
            height0  = max(newy)-miny
            newy = (newy-miny) *height/height0 + dy_this1 
            contoursy[i] = newy
            pass

        pass
        return  contoursx, contoursy
   
    def random_shape_contour(H,W,x,y):
        # first need to determine whether use the origina lcountour to shift 
        dc1 =np.random.random_sample()*10
        dc1  = int(dc1)%2
        if dc1==0: # use the original signal 
            # inital ramdon width and height
           
            width =  int((0.05+0.91* np.random.random_sample())*W)
            dx1 = int(  np.random.random_sample()*(W-width)  )
            dx2  = dx1+width
            dy1 = int(  np.random.random_sample()*H*1.5 -0.25*H)
            dy2  = int  ( np.random.random_sample()*(H*1.5-dy1)) + dy1

            height =  dy2-dy1
            # star and end
            #new x
            newx = np.arange(dx1, dx2)
            #new y based on a given original y
            newy=signal.resample(y, width)
            r_vector   = np.random.sample(20)*50
            r_vector=signal.resample(r_vector, width)
            r_vector = gaussian_filter1d (r_vector ,10)
            newy = newy + r_vector
            miny=min(newy)
            height0  = max(newy)-miny
            newy = (newy-miny) *height/height0 + dy1 
            newy = np.clip(newy,0,H-1)
        else:       
            newy = y
            newx = x
        #width  = 30% - % 50


        #sample = np.arange(width)
        #r_vector   = np.random.sample(20)*20
        #r_vector = gaussian_filter1d (r_vector ,10)
        #newy = np.sin( 1*np.pi/width * sample)
        #newy = -new_contoury*(dy2-dy1)+dy2
        #newy=new_contoury+r_vector
        #newx = np.arange(dx1, dx2)
        return newx,newy
    #draw color contour 
    def add_noise_or_not(img):
        noise_selector=['speckle','s&p','gauss_noise','gauss_noise']
        noise_it = np.random.random_sample()*5
        noise_type1  =  str( noise_selector[int(noise_it)%4])  
        img  =  Basic_Operator.noisy(noise_type1,img)
        return img

    def noisy(noise_typ,image):
           if noise_typ == "none":
              return image
           if noise_typ == "gauss_noise":
              row,col = image.shape
              mean = 0
              var = 50
              sigma = var**0.5
              gauss = np.random.normal(mean,sigma,(row,col )) 
              gauss = gauss.reshape(row,col ) 
              noisy = image + gauss
              return np.clip(noisy,0,254)
           elif noise_typ == 's&p':
              row,col  = image.shape
              s_vs_p = 0.5
              amount = 0.004
              out = np.copy(image)
              # Salt mode
              num_salt = np.ceil(amount * image.size * s_vs_p)
              coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
              out[coords] = 1

              # Pepper mode
              num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
              coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
              out[coords] = 0
              return np.clip(out,0,254)
           elif noise_typ == 'poisson':
              vals = len(np.unique(image))
              vals = 2 ** np.ceil(np.log2(vals))
              noisy = np.random.poisson(image * vals) / float(vals)
              return np.clip(noisy,0,254)
           elif noise_typ =='speckle':
              row,col  = image.shape
              gauss = np.random.randn(row,col )
              gauss = gauss.reshape(row,col )        
              noisy = image + image * gauss
              return np.clip(noisy,0,254)

    def ramdom_speckle(img):
        H,W = img.shape 
        mask = np.zeros((H,W))
        pos_vec =  (np.random.sample(20)*0.8+0.1)*W
        for pos in  pos_vec:
            max = np.random.random_sample()*100+150
            min = 0
            endpoint = int(np.random.random_sample() * H)
            line = np.linspace(max, min, num=endpoint)
            mask[0:endpoint,int(pos)]=mask[0:endpoint,int(pos+1)]  = line
        #smooth the mask 
        mask =  cv2.blur(mask,(5,5))
        img = img + mask
        return img
    def add_speckle_or_not (img):
        Dice = int( np.random.random_sample()*10)
        if Dice % 1 ==0 :
            return Basic_Operator.ramdom_speckle(img)

        else:
            return img

    def ramdom_gap(img):
        H,W = img.shape 
        mask = np.ones((H,W))
        pos_vec = (np.random.sample(20)*0.8+0.1)*W
        for pos in  pos_vec:
            #max = np.random.random_sample()*50+180
            #min = 0
            #endpoint = int(np.random.random_sample() * H)
            #line = np.linspace(max, min, num=endpoint)
            mask[:,int(pos)] *= 0
            mask[:,int(pos+1)] *= 0
            mask[:,int(pos-1)] *= 0


        #smooth the mask 
        mask =  cv2.blur(mask,(5,5))
        img = np.multiply(img,mask)
        return img
    def add_gap_or_not (img):
        Dice = int( np.random.random_sample()*10)
        if Dice % 1 ==0 :
            return Basic_Operator.ramdom_gap(img)

        else:
            return img

        
    def draw_coordinates_color(img1,vx,vy,color):       
            if color ==0:
               painter  = [254,0,0]
            elif color ==1:
               painter  = [0,254,0]
            elif color ==2:
               painter  = [0,0,254]
            else :
               painter  = [0,0,0]
                        #path0  = signal.resample(path0, W)
            H,W,_ = img1.shape
            for j in range (len(vx)):
                    #path0l[path0x[j]]
                    dy = np.clip(vy[j],2,H-2)
                    dx = np.clip(vx[j],2,W-2)
                    img1[int(dy)+1,dx,:]=img1[int(dy)-1,dx,:]=img1[int(dy),dx,:]=painter
            return img1
    def gray2rgb(img):
        new=np.zeros((img.shape[0],img.shape[1],3))
        new[:,:,0]  = img
        new[:,:,1]  = img
        new[:,:,2]  = img

        return new
    def warp_padding(img,contour0,new_contour):
        shift_vector =  new_contour  - contour0
        new_image  = img
        for iter in range(len(shift_vector)):
            lineshift= int(shift_vector[iter] )
            new_image[:,iter] =np.roll( img[:,iter] ,lineshift)
            # The carachter within the contour need special stratigies to maintain 
            if(lineshift>0):#
                origin_point  = int(contour0[iter])
                new_image[0:lineshift,iter]= signal.resample(img[0:origin_point,iter], 
                                                             lineshift)
                pass
        return new_image
        #just roll one line 
    def warp_padding_line1(line,y0,new_y):
        shift  =  int(new_y  - y0)
        new_y = int(new_y)
        y0 = int (y0)
        line_new =np.roll( line ,shift)
        # The carachter within the contour need special stratigies to maintain 
        if(shift>0):#
            line_new[0:new_y]= signal.resample(line[0:y0],new_y)           
        return line_new
    def warp_padding_line2(line,y0,new_y):
        shift  =  int(new_y  - y0)
        new_y = int(new_y)
        y0 = int (y0)
        line_new =np.roll( line ,shift)
        # The carachter within the contour need special stratigies to maintain 
        if(shift>0):#
            line_new[0:new_y]= signal.resample(line[int(y0/2):y0],new_y)           
        return line_new

    # a paathech
    def generate_patch_with_contour(img1,H_new,contour0x,contour0y,
                                    new_contourx,new_contoury):
        H,W  = img1.shape
        img1 = cv2.resize(img1, (W,H_new), interpolation=cv2.INTER_AREA)
        contour0y = contour0y*H_new/H
        points = len(contour0x)
        points_new = len(new_contoury)
        W_new  = points_new
        new  = np.zeros((H_new,W_new))
        for i in range(W_new):
            line_it = int( np.random.random_sample()*points)
            line_it = np.clip(line_it,0,points-1) 
            source_line = img1[:,contour0x[line_it]]
            #new[:,i] = ba.warp_padding_line1(source_line, contour0y[line_it],new_contoury[i])
            new[:,i] = Basic_Operator .warp_padding_line2(source_line, contour0y[line_it],new_contoury[i])
            #random select a source
            pass
        pass
        return new
    def generate_patch_base_origin(img1,H_new,contour0x,contour0y,
                                    new_contourx,new_contoury):
        H,W  = img1.shape
        contour0y = contour0y*H_new/H
        points = len(contour0x)
        points_new = len(new_contoury)
        W_new  = points_new
        original_patch  =  img1[:,contour0x[0]:contour0x[points-1]]
        original_patch  =  cv2.resize(original_patch, (points_new,H_new), interpolation=cv2.INTER_AREA)
        contour0y=signal.resample(contour0y, W_new)
        img1 = cv2.resize(img1, (W,H_new), interpolation=cv2.INTER_AREA)
        
        new  = np.zeros((H_new,W_new))
        for i in range(W_new):
            #line_it = int( np.random.random_sample()*points)
            #line_it = np.clip(line_it,0,points-1) 
            source_line = original_patch[:,i]
            #new[:,i] = ba.warp_padding_line1(source_line, contour0y[line_it],new_contoury[i])
            new[:,i] = Basic_Operator .warp_padding_line2(source_line, contour0y[i],new_contoury[i])
            #random select a source
            pass
        pass
        return new
        #fill in a long vector with a small er on e
    #this bversion just use the resample
    #next version can use the clone
    def fill_lv_with_sv1(sv,H):
        #h = len(sv)
        #div = (H/h)
        #if div > 2:
        #    for i in range(3):
        #        sv=np.append(sv,sv)
        lv=signal.resample(sv, H)     
        return lv
    def fill_lv_with_sv2(sv,H):
        lv = np.zeros(H)
        h = len(sv)
        div = (H/h)
        div= int(math.log2(div))+1
        for i in range(div):
            sv=np.append(sv,sv)
        lv = sv[0:H] 
        return lv
    # to generate synthetic background with this image with label 
    def generate_background_image2(img,contourx,contoury,H,W):
        ori_H,ori_W  = img.shape
        points = len(contourx)
        new  = np.zeros((H,W))
        c_len = len(contourx)
        Dice = int( np.random.random_sample()*10)
        #method 1 just use the left and right side of the imag to raasampel
        if Dice % 2 ==0 and c_len<0.6* ori_W:
            sourimg1  = img[:,0:contourx[0]]
            sourimg2  = img[:,contourx[c_len-2]: ori_W]
            sourimg = np.append(sourimg1,sourimg2,axis =1)
            new  = cv2.resize(sourimg, (W,H), interpolation=cv2.INTER_AREA)
        else:
            #method 2 the line is generated with the line above the the contour 
            #generate line by line 
            for i in range(W):
                #random select a source
 
                #random select a A-line
                line_it = int( np.random.random_sample()*points)
                line_it = np.clip(line_it,0,points-1) 
                y = contoury[line_it]
                #pick part of the A-line betwenn contour and scanning center
                source_line = img[int(0.3*y):int(0.6*y),contourx[line_it]]

                #source_h  = h_list[pic_it]
                #new[:,i] = self.fill_lv_with_sv1(source_line,H)
                new[:,i] =  Basic_Operator .fill_lv_with_sv1(source_line,H)
        return new