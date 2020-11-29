import numpy as np

class EKF(object):
    def __init__(self,delta_x):
        ## init based on the warping vector lenth

        l = len(input) # the length of all the states
        self.l = l
        self.x = np.zeros(l) # init the state
        self.P = np.identity(l)*10 # the uncertanty 
        #self.P = np.matmul(self.P,self.P)
        self.P =  self.P @ self.P 

        self.Q = np.identity(l)*0.1
        self.R = np.identity(l)*3
        self.I = np.identity(l)
        pass
    def predict(self):
        self.xbar = self.x + self.delta_x
        ###compute the transation matrix
        self.F= self.I
        #self.P = F @ self.P @ F.transpose() + self.Q #predict P
        self.P =  self.P  + self.Q #predict P


        pass
    def correct(self):
        self.z = np.zeros(self.l) + self.r # overall rotation is the observation
        self.zhat = self.x   # it is Identity mapping 
        self. H  = self.I
        #self.K = (self.P @ self.H.transpose())/(self.H @ self.P @ self.H.transpose() + self.R)
        InvPR = np.linalg.inv(self.P   + self.R)
        self.K = (self.P ) @ (InvPR)
        # the dot devide should be the same  for dialog matix
        #so
        #self.K = (self.P )/(self.P   + self.R)

        self.x = self.xbar + self.K @ (self.z-self.zhat)
        #self.P = (self.I - self.K*H)*ekfP
        self.P = (self.I - self.K )*self.P

        
        pass
    def update(self,delta_x,r): # input the warping vector, and group rotation value 
        self.delta_x = delta_x
        self.r = r
        self.predict()
        self.correct()
        pass
        return self.x

if __name__ == '__main__':
    p = np.zeros(832)
    sekf1 = EKF(,p)




