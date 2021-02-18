from visdom import Visdom
import numpy as np

#  this is  the online plotter which only need input on dot of the line in one time

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
        self.previous_title =[]
        self.legend = []
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name ,
                xlabel='Epochs',
                ylabel= title_name 
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
    def plot_multi_arrays_append(self,x,y,title_name, legend):
         if self.previous_title == title_name:
             self.legend.append(legend)
         else:
             self.legend = [legend]
         self.previous_title = title_name
         self.viz.line(
            X=x,
            Y=y,
            win=title_name,
            name=legend,
            update='append',
           opts={
             'title' : title_name,
          
             'legend':  self.legend,
            #'dash': np.array(['dot']),
                     }
                 )
 



# test  the offline plot which use the vis.line function to inout the matrixc or the array in omne time 
if __name__ == '__main__':

     # the demo of plot multiple figures in one window, if the windwo
     # name is the sam e the new figure will be add the  same window 
   
        ploter2 = VisdomLinePlotter()
        y = np.random.rand(10)*100
        y2 = np.random.rand(10)*100
        y3= np.random.rand(10)*100


        x=np.arange(0, len(y))
        ploter2.plot_multi_arrays_append(x,y,title_name='test',legend = 'origin' )
        ploter2.plot_multi_arrays_append(x,y2,title_name='test',legend = 'deep' )
        ploter2.plot_multi_arrays_append(x,y3,title_name='test',legend = 'tradition' )

      
        #vis.line(Y=np.random.rand(10)*100,    
        #         opts=dict(xlabel='X',
        #                ylabel='Warping value',
        #                title='Training Loss',
        #                legend=['Loss']),
        #         update='insert'
                 
        #         )
        #vis= Visdom()
        #vis.line(
        #        X=np.arange(1, 38),
        #        Y=np.random.randn(37),
        #        win="test3",
        #        name='6',
        #        update='append',
        #        )
        #vis.line(
        #    X=np.arange(1, 38),
        #    Y=np.random.randn(37),
        #    win="test3",
        #    name='11',
        #    update='append',
        #    )
        #loss_window = vis.line(X=torch.zeros((1,)).cpu(),
        #                       Y=torch.zeros((1)).cpu(),
        #                       opts=dict(xlabel='epoch',
        #                                 ylabel='Loss',
        #                                 title='Training Loss',
        #                                 legend=['Loss']))    
        #vis =  Visdom()
        #accuracy_window = vis.line(X=torch.zeros((1,)).cpu(),
        #                       Y=torch.zeros((1)).cpu(),
        #                       opts=dict(xlabel='epoch',
        #                                 ylabel='accuracy',
        #                                 title='Training accuracy',
        #                                 legend=['accuracy']))    
        
        
        #for epoch in range(200):  

        #    vis.line(
        #                X=torch.ones(5).cpu(),
        #                Y=torch.Tensor([1,2,3,4,5]).cpu(),
        #                win=loss_window,
        #                update='append')
      
        pass

