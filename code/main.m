clc,clear
load('..\user_data\Configuration.mat');

for N_Data = 11:11 % {'x264','SQL','sac','LLVM','javagc','hsmgp','hipacc','Dune','BDBJ','BDBC','Apache'}
    [X, Y, CluRe, DataSetName, mf] = Setup(N_Data);
    SCORE = csvread(['..\user_data\result_',DataSetName,'.csv']);% Read log file
    if N_Data == 3||N_Data == 5||N_Data == 6||N_Data == 7||N_Data == 8 %Sampling number setting
        k = 8;
        p = 7;
    else       
        k = 2 * A(2,N_Data);
        p = 2 * A(2,N_Data);
    end
    for Experiment_number = 1:1
        tic
        
        fprintf([DataSetName, ': epoch = ', num2str(Experiment_number),' Running...']);
        [model,mre,N,M] = RSFIN(X,Y,mf,CluRe,k,p,A(1,N_Data));   
        fprintf(['\b\b\b\b\b\b\b\b\b\bFinish\n','mre = ', num2str(mre),...
            ' Time cost: ',num2str(toc),' s\n']);
        information_print(model, DataSetName);
        
        SCORE(Experiment_number, 1) = Experiment_number;
        SCORE(Experiment_number, 2) = mre;
        SCORE(Experiment_number, 3) = N;
        SCORE(Experiment_number, 4) = size(model.MR,2);
        SCORE(Experiment_number, 5) = toc;
        csvwrite(['..\user_data\result_',DataSetName,'.csv'],SCORE);
    end  
end

disp(['maen = ', num2str(mean(SCORE(:,2))),...
      ' Margin = ',num2str(1.96*std(SCORE(:,2))/sqrt(SCORE(Experiment_number, 3))),...
      ' Time(per epoch) = ',num2str(mean(SCORE(:,5)))]);

%% boxplot
Data = [];
for N_Data = 1:11
    if N_Data == 5|| N_Data == 3
        continue;
    end
    [X, Y, CluRe, DataSetName, mf] = Setup(N_Data); % {'x264','SQL','sac','LLVM','javagc','hsmgp','hipacc','Dune','BDBJ','BDBC','Apache'}
    SCORE = csvread(['..\user_data\result_',DataSetName,'.csv']);
%     SCORE = csvread(['..\user_data\result_Twolayer_',DataSetName,'.csv']);
    
    Data = [Data,  SCORE(:,2)];
end
notBoxPlot(Data,'jitter',0.6);