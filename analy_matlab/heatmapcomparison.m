% open the mat file in in the paper1 / matlab_fig/mat_MSE

MAP=[
     DeepVas, DeepLung,DeepDige,DeepRect,DeepPhan;
     GSVas, GSLung,GSDige,GSRect,GSPhan;
    DeepVasS, DeepLungS,DeepDigeS,DeepRectS,DeepPhanS;
     GSVasS, GSLungS,GSDigeS,GSRectS,GSPhanS;
    ]

Tra = [ GSVas, GSLung,GSDige,GSRect,GSPhan,GSVas, GSLung,GSDige,GSRect,GSPhan]
Deep= [DeepVas, DeepLung,DeepDige,DeepRect,DeepPhan,DeepVasS, DeepLungS,DeepDigeS,DeepRectS,DeepPhanS ] 
