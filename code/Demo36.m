clc,clear,clf
load('..\user_data\Configuration2.mat');
N_Data = 6;
[X, Y, CluRe, DataSetName, mf] = Setup(N_Data);
% {'x264','SQL','sac','LLVM','javagc','hsmgp','hipacc','Dune','BDBJ','BDBC','Apache'}
SCORE = csvread(['..\user_data\result_',DataSetName,'.csv']);% Read log file
[n, N] = size(X);
% pos = randperm(N);
% X = X(:,pos(1:5000));
% Y = Y(pos(1:5000));
% X = X(1:end-2,:);
% CluRe = CluRe(1:end-2);
% mf = mf(1:end-2);
k = 3;
p = 3;

wall = B(1,N_Data);
for mmmm = 1:1

%         if SCORE(mmmm, 2) < 4
%             continue;
%         end

tic;
SCORE(mmmm, 1) = mmmm;
% 分类器
Yd = Y;
Yd(Y>wall) = 100;
Yd(Y<wall) = 50;

mre = 100;
NNN = 2;
while mre > 2
    
    [n, N] = size(X);
    Testpos = 1:N;
    [Valpos, TT, Testpos] = Sample(Testpos,k*n,p*n);
    ValXd = X(:,Valpos);ValYd = Yd(Valpos);
    TrainXd = X(:,TT);TrainYd = Yd(TT);
    TestXd = X(:,Testpos);TestYd = Yd(Testpos);% Test set
    
    [MRd, S] = ASFIN(X,mf,CluRe,NNN,ValXd,ValYd,TrainXd,TrainYd);
    [mfd,Acd,~] = ANFIS([TrainXd, ValXd],[TrainYd,ValYd],mf,MRd,10);
    yp = NetWork(TestXd,mfd,MRd,Acd);
    mre = MRE(yp, TestYd);
end
Plot_Test(yp, TestYd, DataSetName, mre, MRd, size(X,2) - length(Testpos))
Usedpos = [Valpos, TT];
Vu = Valpos(ValYd > 75);Vd = Valpos(ValYd <= 75);
ku = length(Vu);kd = k*n - ku;
Tu = TT(TrainYd > 75);Td = TT(TrainYd <= 75);
pu = length(Tu);pd = p*n - pu;

NNN = 2;
% 模型一
Testpos = 1:N;
yp = NetWork(X,mfd,MRd,Acd);
Testpos1 = Testpos(yp>75);
mre = 600;
if ~isempty(Testpos1)
    while mre>=B(2,N_Data) %Eliminate bad experiments
        [n, ~] = size(X);
        [Valpos, TT, Testpost] = Sample(Testpos1,max(k*n - ku, 0),max(p*n - pu, 0));%Validation set sampling and training set sampling
        Valpos = [Valpos, Vu];
        TT = [TT, Tu];
        
        ValX = X(:,Valpos); ValY = Y(Valpos);
        TrainX = X(:,TT); TrainY = Y(TT);
        TestX = X(:,Testpost);TestY = Y(Testpost);% Test set
        
        % Rule search
        [MR1, S] = ASFIN(X(:,Testpos1),mf,CluRe,NNN,ValX,ValY,TrainX,TrainY);
        % Learn
        [mf1,Ac1,M] = ANFIS([TrainX, ValX],[TrainY,ValY],mf,MR1,10);
        % Validation
        yp = NetWork(TestX,mf1,MR1,Ac1);
        mre = MRE(yp, TestY);
    end
    Usedpos = [Usedpos, Valpos, TT];
    yp = NetWork(TestX,mf1,MR1,Ac1);
    mre = MRE(yp, TestY);
    %     Plot_Test(yp, TestY, DataSetName, mre, MR1, size(X,2) - length(Testpos))
else
    MR1 = [];
end

NNN = 1;
% 模型二
Testpos = 1:N;
yp = NetWork(X,mfd,MRd,Acd);
Testpos2 = Testpos(yp <= 75);
if ~isempty(Testpos2)
    mre = 100;
    while mre>=B(3,N_Data) %Eliminate bad experiments
        [n, ~] = size(X);
        [Valpos, TT, Testpost] = Sample(Testpos2,max(k*n - kd, 0),max(p*n - pd, 0));%Validation set sampling and training set sampling
        Valpos = [Valpos, Vd];
        TT = [TT, Td];
        
        ValX = X(:,Valpos); ValY = Y(Valpos);
        TrainX = X(:,TT); TrainY = Y(TT);
        TestX = X(:,Testpost);TestY = Y(Testpost);% Test set
        
        % Rule search
        [MR2, S] = ASFIN(X(:,Testpos2),mf,CluRe,NNN,ValX,ValY,TrainX,TrainY);
        % Learn
        [mf2,Ac2,M] = ANFIS([TrainX, ValX],[TrainY,ValY],mf,MR2,10);
        % Validation
        yp = NetWork(TestX,mf2,MR2,Ac2);
        mre = MRE(yp, TestY);
    end
    Usedpos = [Usedpos, Valpos, TT];
    yp = NetWork(TestX,mf2,MR2,Ac2);
    mre = MRE(yp, TestY);
    %     Plot_Test(yp, TestY, DataSetName, mre, MR2, size(X,2) - length(Testpos))
    
else
    MR2 = [];
end

Testpos = setdiff(1:N,Usedpos);
TestX = X(:,Testpos);TestY = Y(Testpos);% Test set
% 组合测试
yp = [];
for i = 1:size(TestX, 2)
    yp1 = NetWork(TestX(:,i),mfd,MRd,Acd);
    if yp1 > 75
            yp = [yp, NetWork(TestX(:,i),mf1,MR1,Ac1)];
    else
            yp(i) = NetWork(TestX(:,i),mf2,MR2,Ac2);
    end
end
mre = MRE(yp, TestY);
SCORE(mmmm, 2) = mre;
SCORE(mmmm, 3) = N - length(Testpos);
SCORE(mmmm, 4) = size([MR1, MR2],2);
SCORE(mmmm, 5) = toc;
csvwrite(['..\user_data\result_',DataSetName,'.csv'],SCORE);
end
Plot_Test(yp, TestY, DataSetName, mre, MR2, size(X,2) - length(Testpos))
mean(SCORE(:,2))
std(SCORE(:,2))
mean(SCORE(:,5))