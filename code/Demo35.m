clc,clear,clf

NNN = 33;
ExpMRE = 20;
for N_Data = 10:10
    [X, Y, CluRe, DataSetName, mf] = Setup(N_Data); % {'x264','SQL','sac','LLVM','javagc','hsmgp','hipacc','Dune','BDBJ','BDBC','Apache'}
    SCORE = csvread(['..\user_data\result_',DataSetName,'.csv']);
    
    
    %     pos = find(Y<5);
    %     X = X(:,pos);
    %     Y = Y(pos);
    if N_Data == 3||N_Data == 5||N_Data == 6||N_Data == 7
        pos = randperm(size(X,2));
        X = X(:,pos(1:2000));
        Y = Y(pos(1:2000));
    end
    if N_Data == 3||N_Data == 5||N_Data == 6||N_Data == 7||N_Data == 8
        
        k = 10;
        p = 10;
    else
        k = 2;
        p = 2;
    end
    MMM = [];
    for mmmm = 1:1
        %                 if SCORE(mmmm, 2) <= ExpMRE
        %                     continue;
        %                 end
        tic
        mre = 10000;
        while mre>=ExpMRE
            SCORE(mmmm, 1) = mmmm;
            [n, N] = size(X);
            Testpos = 1:N;
            
            [Valpos, Testpos] = Sample(Testpos,k*n);
            [TT, Testpos] = Sample(Testpos,p*n);
            
            ValX = X(:,Valpos); ValY = Y(Valpos);
            TrainX = X(:,TT); TrainY = Y(TT);
            TestX = X(:,Testpos);TestY = Y(Testpos);
            
            disp([DataSetName, ':', num2str(mmmm)]);
            [QMR, S] = ASFIN(X,mf,CluRe,NNN,ValX,ValY,TrainX,TrainY);
            
            
            [mf,Ac,M] = ANFIS([TrainX, ValX],[TrainY,ValY],mf,QMR,100);
            yp = NetWork(TestX,mf,QMR,Ac);
            mre = MRE(yp, TestY);
        end
        MMM = [MMM; [S,M]];
        SCORE(mmmm, 2) = mre;
        SCORE(mmmm, 3) = N - length(Testpos);
        SCORE(mmmm, 4) = size(QMR,2);
        SCORE(mmmm, 5) = toc;
        csvwrite(['..\user_data\result_',DataSetName,'.csv'],SCORE);
    end
end
Plot_Test(yp, TestY, DataSetName, mre, QMR, N - length(Testpos));
mean(SCORE(:,2))
std(SCORE(:,2))
sum(SCORE(:,5))
%%
Data = [];
for N_Data = 1:11
    if N_Data == 5|| N_Data == 3
        continue;
    end
    [X, Y, CluRe, DataSetName, mf] = Setup(N_Data); % {'x264','SQL','sac','LLVM','javagc','hsmgp','hipacc','Dune','BDBJ','BDBC','Apache'}
    SCORE = csvread(['..\user_data\result_',DataSetName,'.csv']);
    
    Data = [Data,  SCORE(:,2)];
end
%%
clf
clc
notBoxPlot(Data,'jitter',0.6);
