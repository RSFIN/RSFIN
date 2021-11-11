clc,clear
load('..\user_data\Configuration2.mat');
for N_Data = 10:10
    [X, Y, CluRe, DataSetName, mf] = Setup(N_Data); % {'x264','SQL','sac','LLVM','javagc','hsmgp','hipacc','Dune','BDBJ','BDBC','Apache'}
    SCORE = csvread(['..\predicton_result\result_TwoLayer_',DataSetName,'.csv']);% Read log file
    
    if N_Data >= 1 && N_Data <= 11
        wall = B(1,N_Data);
        ExpMRE0 =B(2,N_Data);
        ExpMRE1 = B(3,N_Data);
        ExpMRE2 = B(4,N_Data);
    else
        wall = mean(Y);
        ExpMRE0 = 20;
        ExpMRE1 = 50;
        ExpMRE2 = 50;
    end
    for mmmm = 30:30

        tic
        fprintf([DataSetName, ': epoch = ', num2str(mmmm),' Running...']);
        [modelTwo,mre,N] = TwoLayerRSFIN(X,Y,mf,CluRe,9/8,9/8,9/8,9/8,wall,ExpMRE0,ExpMRE1,ExpMRE2);
        fprintf(['\b\b\b\b\b\b\b\b\b\bFinish\n','mre = ', num2str(mre),...
            ' Time cost: ',num2str(toc),' s\n']);
        SCORE(mmmm, 1) = mmmm;
        SCORE(mmmm, 2) = mre;
        SCORE(mmmm, 3) = N;
        SizeMR = 0;
        if ~isempty(modelTwo.model_1)
            SizeMR = SizeMR + size(modelTwo.model_1.MR,2);
        end
        if ~isempty(modelTwo.model_2)
            SizeMR = SizeMR + size(modelTwo.model_2.MR,2);
        end
        SCORE(mmmm, 4) = SizeMR;
        SCORE(mmmm, 5) = toc;
        csvwrite(['..\predicton_result\result_TwoLayer_',DataSetName,'.csv'],SCORE);
    end
end

disp(['maen = ', num2str(mean(SCORE(:,2))),...
    ' Margin = ',num2str(1.96*std(SCORE(:,2))/sqrt(SCORE(mmmm, 3))),...
    ' Time(per epoch) = ',num2str(mean(SCORE(:,5)))]);