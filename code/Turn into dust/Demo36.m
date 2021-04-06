clc,clear,clf

for N_Data = 1:1 %3~10
    [X, Y, CluRe, DataSetName, mf] = Setup(N_Data); % {'x264','SQL','sac','LLVM','javagc','hsmgp','hipacc','Dune','BDBJ','BDBC','Apache'}
    SCORE = csvread(['..\user_data\result_',DataSetName,'.csv']);
%     pos = randperm(size(X,2));
%         X = X(:,pos(1:2000));
%         Y = Y(pos(1:2000));
    for mmmm = 1:1
        %         if SCORE(mmmm, 2) ~= 0
        %             continue;
        %         end
        tic
        mre = 10000;
        while mre>=100
            
            SCORE(mmmm, 1) = mmmm;%序号
            UsedPos = [];
            [n, N] = size(X);
            Testpos = 1:size(X,2);
            
            if N_Data == 3||N_Data == 5||N_Data == 6||N_Data == 7||N_Data == 8 %采样数量
                k = 10;
                p = 10;
            else
                k = 4;
                p = 1;
            end
            
            QMR = [];
            [Valpos, Testpos] = Sample(Testpos,k*n);%采样
            ValX = X(:,Valpos); ValY = Y(Valpos);
            
            CluRes = MeanShift(ValY);
            CluY = CluRes{1};
            TT = [];
            for i = 1:length(CluY)
                [~, Best] = min(abs(ValY - CluY(i)));
                MR = X2MR(ValX(:,Best(1)),mf);
                [~, pos] = sort(O2(X(:,Testpos),mf,MR), 'descend');
                TT = [TT, pos(2:5)];
                QMR = [QMR, MR];
            end
            
            TrainX = X(:,TT); TrainY = Y(TT);
            Testpos = setdiff(Testpos, TT);
            
            flag = 0;
            tmre = 100;
            for asd = 1:100
                
                if flag >= 10
                    break;
                end
                AMR = QMR;
                
                
                MR = GenMR(AMR,mf,CluRe,TrainX);
                AMR = [AMR,MR];
                
                [socer, S] = SocerOfMR(X(:,randi(size(X,2),100,1)),mf,AMR);
                
                if ~isempty(AMR(:, socer > 0.9))
                    AMR(:, socer < 0.9) = [];
                end
                AMR = Union(AMR);
                
                
                socer = SocerOfMR(X(:,randi(size(X,2),100,1)),mf,AMR);
                
                [mf,~] = ANFIS(TrainX,TrainY,mf,AMR,1);
                temp = size(AMR,2);
                if ~isempty(AMR(:, socer > 0.9))
                    AMR(:, socer < 0.9) = [];
                end
                
                [mf,Ac] = ANFIS(TrainX,TrainY,mf,AMR,1);
                
                yp = NetWork(ValX,mf,AMR,Ac);
                mre = MRE(yp, ValY);
                flag = flag + 1;
                if mre < tmre
                    flag = 0;
                    tmre = mre;
                    QMR = AMR;
                end
                disp([DataSetName, ':', num2str(mmmm),', ',...
                    num2str(asd),'/100, mre: ',num2str(tmre)]);
            end
            
            TestX = X(:,Testpos);TestY = Y(Testpos);
            [mf,Ac] = ANFIS([TrainX, ValX],[TrainY,ValY],mf,QMR,10);
            yp = NetWork(TestX,mf,QMR,Ac);
            mre = MRE(yp, TestY);
        end
        %         SCORE(mmmm, 2) = mre;
        %         SCORE(mmmm, 3) = length(unique(UsedPos));
        %         SCORE(mmmm, 4) = size(Union(QMR),2);
        %         SCORE(mmmm, 5) = toc;
        %         csvwrite(['..\user_data\result_',DataSetName,'.csv'],SCORE);
    end
end
Plot_Test(yp, TestY, DataSetName, mre, Union(QMR), size(X,2) - length(Testpos));
% boxplot(SCORE(:,2));
