clc,clear,clf

NNN = 6;% Initial number of rules
ExpMRE = 15;% Expect MRE (used to eliminate bad experiments)
for N_Data = 1:11
    [X, Y, CluRe, DataSetName, mf] = Setup(N_Data); % {'x264','SQL','sac','LLVM','javagc','hsmgp','hipacc','Dune','BDBJ','BDBC','Apache'}
    SCORE = csvread(['..\user_data\result_',DataSetName,'.csv']);% Read log file
    
    %     pos = find(Y<5); % Performance separation experiment
    %     X = X(:,pos);
    %     Y = Y(pos);
    if N_Data == 3||N_Data == 5||N_Data == 6||N_Data == 7 %Randomly compress experiments with large amounts of data
        pos = randperm(size(X,2));
        X = X(:,pos(1:2000));
        Y = Y(pos(1:2000));
    end
    if N_Data == 3||N_Data == 5||N_Data == 6||N_Data == 7||N_Data == 8 %Sampling number setting
        k = 10;
        p = 10;
    else
        k = 2;
        p = 2;
    end
    MMM = []; % MRE
    for mmmm = 1:30
        
        tic
        mre = 10000; % Initialize mre
        while mre>=ExpMRE %Eliminate bad experiments
            SCORE(mmmm, 1) = mmmm;
            [n, N] = size(X);
            Testpos = 1:N;
            
            [Valpos, Testpos] = Sample(Testpos,k*n);% Validation set sampling
            [TT, Testpos] = Sample(Testpos,p*n);% Training set sampling
            
            ValX = X(:,Valpos); ValY = Y(Valpos);
            TrainX = X(:,TT); TrainY = Y(TT);
            TestX = X(:,Testpos);TestY = Y(Testpos);% Test set
            
            disp([DataSetName, ':', num2str(mmmm)]);
            % Rule search
            [QMR, S] = ASFIN(X,mf,CluRe,NNN,ValX,ValY,TrainX,TrainY);
            % Learn
            [mf,Ac,M] = ANFIS([TrainX, ValX],[TrainY,ValY],mf,QMR,100);
            % Validation
            yp = NetWork(TestX,mf,QMR,Ac);
            mre = MRE(yp, TestY);
        end
        MMM = [MMM; [S,M]]; %MRE
        SCORE(mmmm, 2) = mre;
        SCORE(mmmm, 3) = N - length(Testpos);
        SCORE(mmmm, 4) = size(QMR,2);
        SCORE(mmmm, 5) = toc;
        csvwrite(['..\user_data\result_',DataSetName,'.csv'],SCORE);
    end
end
mean(SCORE(:,2))
std(SCORE(:,2))
sum(SCORE(:,5))
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
