operatedir  =  '../../saved_matlab/validation_between_frame.mat';
load(operatedir)
evaluate_len =500
draw_each_liNe_flag = 0
for i = 1:evaluate_len
    true = arr.truth(i,:);
    deep = arr.deep_result(i,:);
    tradition = arr.tradition_result(i,:);
%     Mean Square Error
    MSE_deep(i)=sum(abs((true-deep).^2))/length(true);
    MSE_tradition(i)=sum(abs((true-tradition).^2))/length(true);
    if draw_each_liNe_flag ==1
        figure(1)
        hold off;
        plot(true);hold on
        plot(deep);hold on
        plot(tradition);hold on
        grid on
        legend('true','deep','tradition')
    end
    
end
figure(2)
hold off;
plot(MSE_deep);hold on
plot(MSE_tradition);hold on
% plot(tradition);hold on
grid on
legend('deep','tradition')
% image = imread(operatedir);
% figure(2)
% imshow (image);
% a= x16(:,1:50)
% vector = sum(a,2)
% new  = x29(:,1:10:774)
 