 
fig = gcf;
dataObjs = findobj(fig,'-property','YData');
 MSE_tradition= dataObjs(1).YData;
MSE_deep = dataObjs(2).YData;