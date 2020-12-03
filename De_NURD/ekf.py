import numpy as np

class EKF(object):
    def __init__(self ):
        ## init based on the warping vector lenth

        l = 832 + 1  # the length of all the states the last states is the bias of warping vector
        self.l = l
        self.x = np.zeros(l) # init the state
        self.P = np.identity(l)*1 # the uncertanty 
        #self.P = np.matmul(self.P,self.P)
        self.P =  self.P @ self.P 

        self.Q = np.identity(l)*0.001
        self.R = np.identity(l-1)*0.01
        self.I = np.identity(l)
        pass
    def predict(self):
        self.xbar = self.x + np.append(self.delta_x - self.x[self.l-1],0) # update with x and its bias
        ###compute the transation matrix
        self.F= self.I
        self.F[:,self.l-1] = -1  # the last colum is -1 due to bias 
        self.F[self.l-1,self.l-1] = 0 # the differentioal of bias is zero
        self.P = self.F @ self.P @ self.F.transpose() + self.Q #predict P
        #self.P =  self.P  + self.Q #predict P


        pass
    def correct(self):
        #obsevation has one less dimention
        self.z = np.zeros(self.l-1) + self.r # overall rotation is the observation
        self. H  = np.append( np.identity(self.l-1) , np.zeros((self.l-1,1)),1)

        #self.zhat = self.H @ self.x   # it is Identity mapping 
        self.zhat = self.H @ self.xbar  # it is Identity mapping 

        #self.K = (self.P @ self.H.transpose())/(self.H @ self.P @ self.H.transpose() + self.R)
        InvPR = np.linalg.inv(self.H @ self.P @ self.H.transpose() + self.R)
        self.K = (self.P @ self.H.transpose() ) @ (InvPR)
        # the dot devide should be the same  for dialog matix
        #so
        #self.K = (self.P )/(self.P   + self.R)

        self.x = self.xbar + self.K @ (self.z-self.zhat)
        self.P = (self.I - self.K@ self.H)@ self.P
        #self.P = (self.I - self.K )*self.P

        
        pass
    def update(self,delta_x,r): # input the warping vector, and group rotation value 
        self.delta_x = delta_x
        self.r = r
        self.predict()
        self.correct()
        pass
        return self.x[0:832]

if __name__ == '__main__':
    p = np.zeros(832)

    ekf1 = EKF(p)
    p = np.ones(832)
    r = 0
    new = ekf1.update(p,r)
    new = ekf1.update(p,r)









