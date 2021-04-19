clc,clear


for N_Data = 1:1
    [X, Y, CluRe, DataSetName, mf] = Setup(N_Data); % {'x264','SQL','sac','LLVM','javagc','hsmgp','hipacc','Dune','BDBJ','BDBC','Apache'}
    SCORE = csvread(['..\user_data\result_',DataSetName,'.csv']);% Read log file
    if N_Data == 3||N_Data == 5||N_Data == 6||N_Data == 7||N_Data == 8 %Sampling number setting
        k = 5;
        p = 5;
    else
        k = 2;
        p = 2;
    end
    MMM = []; % MRE
    for mmmm = 1:1
        tic
        fprintf([DataSetName, ': epoch = ', num2str(mmmm),' Running...']);
                
        [mf,Ac,MR,mre,N] = RSFIN(X,Y,mf,CluRe,k,p,N_Data);   
        
        fprintf(['\b\b\b\b\b\b\b\b\b\bFinish\n','mre = ', num2str(mre),...
            ' Time cost: ',num2str(toc),' s\n']);
        
        SCORE(mmmm, 1) = mmmm;
        SCORE(mmmm, 2) = mre;
        SCORE(mmmm, 3) = N;
        SCORE(mmmm, 4) = size(MR,2);
        SCORE(mmmm, 5) = toc;
        csvwrite(['..\user_data\result_',DataSetName,'.csv'],SCORE);
    end
end
% Plot_Test(yp, TestY, DataSetName, mre, QMR, size(X,2) - length(Testpos))
disp(['maen = ', num2str(mean(SCORE(:,2))),...
      ' Margin = ',num2str(1.96*std(SCORE(:,2))/sqrt(SCORE(mmmm, 3))),...
      ' Time(per epoch) = ',num2str(mean(SCORE(:,5)))]);
%% boxplot
Data = [];
for N_Data = 1:11
    if N_Data == 5|| N_Data == 3
        continue;
    end
    [X, Y, CluRe, DataSetName, mf] = Setup(N_Data); % {'x264','SQL','sac','LLVM','javagc','hsmgp','hipacc','Dune','BDBJ','BDBC','Apache'}
    SCORE = csvread(['..\user_data\result_',DataSetName,'.csv']);
    
    Data = [Data,  SCORE(:,2)];
end
notBoxPlot(Data,'jitter',0.6);
