% operatedir  =  'E:\\PhD\\trying\\saved\\'
% image = imread(operatedir);
% figure(2)
% imshow (image);
% a= x16(:,1:50)
% vector = sum(a,2)
% new  = x29(:,1:10:774)
S = size(A);
[X,Y,Z] = ndgrid(1:S(1),1:S(2),1:S(3));
scatter3(X(:),Y(:),Z(:),321,A(:),'filled')

A = rand(50, 50, 50) < 0.01;  % synthetic data
[x y z] = ind2sub(size(A), find(A));
plot3(x, y, z, 'k.');