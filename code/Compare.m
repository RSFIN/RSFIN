clc,clear
DataSetNameSet = {'x264','SQL','LLVM','Apache','BDBC','BDBJ','hipacc','hsmgp','Dune'};
Deep = csvread('box.csv',0,1);
for num = 1:9
       
    DataSetName = DataSetNameSet{num};
    SCORE1 = csvread(['..\user_data\CART_',DataSetName,'_Datails.csv'],91,1);
    SCORE2 = csvread(['..\user_data\result_',DataSetName,'.csv']);
    SCORE3 = csvread(['..\user_data\result_TwoLayer_',DataSetName,'.csv']);

%     Data = [SCORE1(:,5)*100;Deep(num,:)';SCORE2(:,2);SCORE2(:,2)];
    Data = [SCORE1(:,5);Deep(num,:)'];
    
%     for i = 1:length(Data)
%         if i<=30
%             alloy{i} = 'DECART';
%         elseif i<=60
%             alloy{i} = 'DeepPerf';
%         elseif i<=90
%             alloy{i} = 'RSFIN';
%         else
%             alloy{i} = 'TwoLayer RSFIN';
%         end
%     end
    for i = 1:length(Data)
        if i<=30
            alloy{i} = 'RSFIN';
        elseif i<=60
            alloy{i} = 'DeepPerf';
        end
    end
       
        p = kruskalwallis(Data,alloy,'off');
    if p < 0.01
        if mean(Data(1:30))<mean(Data(31:end))
            disp([DataSetName, ': p = ',num2str(round(p,4)),', RSFIN']);
        else
            disp([DataSetName, ': p = ',num2str(round(p,4)),', DeepPerf']);
        end
    else
        disp([DataSetName, ': p = ',num2str(round(p,4)),', No significant difference']);
    end
end