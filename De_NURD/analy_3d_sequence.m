
clc;
close all;
clear all;
test_rate = 0.8;       % ????????
fileFolder=fullfile('.\saved_processed\');
dirOutput=dir(fullfile(fileFolder,'*'));
fileNames={dirOutput.name}';
fullfile = char(fileNames{3:end,:});
test_num    = round(test_rate * length(fullfile));

datadir = [fileFolder,int2str(5),'.jpg'];
test_p = imread(datadir, 'jpg');
A = test_p;
for i = 5 : test_num-5
    datadir = [fileFolder,int2str(i),'.jpg'];
    test_p = imread(datadir, 'jpg');
    A = cat(3,A,test_p);
end
 
[x y z] = ind2sub(size(A/255), find(A/255));
plot3(x, y, z, 'k.');