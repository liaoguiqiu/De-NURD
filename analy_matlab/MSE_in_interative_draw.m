clear
operatedir1  = '../../saved_matlab/infor_shift_NURD.mat' ; % the data of information for generate the video  
operatedir2  ='../../saved_stastics/matlab_sig.mat' ;     % the process result of interative correction 
load(operatedir1)
load(operatedir2)
evaluate_len =500 ;
draw_each_liNe_flag = 0;
% the comparison need to firstly find the start frame(standard frame of )
% of the integral, and calculate the relative 

reference_NURD  = arr.NURD;
refer_shift = arr.overall_shift;
refer_id = arr.label;
Predict = NURD_intergral.path_integral;
Predict_id = NURD_intergral.signals(1,:)

%calculate the reference NURD from the actual starting point
starting = Predict_id(1)% the start name of fram
idx =find(refer_id==starting)
check = refer_id(idx)   % check the name in the reference
%get its position in the reference 
draw_each_liNe_flag = 1
for i = 1:evaluate_len
%       de- bias
      act_refer_nurd = reference_NURD(i+idx-1,:) - reference_NURD(idx-1,:);
%       act_refer_shift = refer_shift(i+idx-1) - refer_shift(idx-1);
      refer_combin = act_refer_nurd ;
      
      deep = 0-Predict(i,:)
        if draw_each_liNe_flag ==1
                figure(1)
                hold off;
                plot(refer_combin);hold on
                plot(deep);hold on
%                 plot(tradition);hold on
                grid on
                legend('true','deep' )
       end
%     true = arr.truth(i,:);
%     deep = arr.deep_result(i,:);
%     tradition = arr.tradition_result(i,:);
%     % Mean Square Error
%     MSE_deep(i)=sum(abs((true-deep).^2))/length(true);
%     MSE_tradition(i)=sum(abs((true-tradition).^2))/length(true);
%     if draw_each_liNe_flag ==1
%         figure(1)
%         hold off;
%         plot(true);hold on
%         plot(deep);hold on
%         plot(tradition);hold on
%         grid on
%         legend('true','deep','tradition')
%     end
    
end