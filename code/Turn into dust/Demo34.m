clc,clear,clf
[X, Y, CluRe, DataSetName, mf] = Setup(11); % {'x264','SQL','sac','LLVM','javagc','hsmgp','hipacc','Dune','BDBJ','BDBC','Apache'}
% pos = randi(size(X,2),1,500);
% X = X(:,pos);
% Y = Y(pos);
UsedPos = [];
[n, N] = size(X);

ValPos = randi(N, 1, 1*n);
TT = [];
AMR = [];
AAc = [];
for i = 1:length(ValPos)
    % GenRule
    MR = X2MR(X(:,ValPos(i)), mf);
    
    % TrainData
    
    [~,TrainPos] = sort(O2(X,mf,MR),'descend');
    TrainPos = TrainPos(2:n);
    TT = [TT,TrainPos];
    AMR = [AMR,MR];
end
TT = unique(TT);
AMR = Union(AMR);

UsedPos = [UsedPos,TT];

%Iteration for train


TrainX = X(:,TT); TrainY = Y(TT);

for i = 1:10
    socer = SocerOfMR(X,mf,AMR);
    
    [mf,~] = ANFIS(TrainX,TrainY,mf,AMR,50);
    temp = size(AMR,2);
    AMR(:,socer<0.9) = [];
    if size(AMR,2) == temp
        break;
    end
end
[mf,Ac] = ANFIS(TrainX,TrainY,mf,AMR,10);

TestX = X;TestY = Y;
TestX(:,UsedPos) = [];
TestY(UsedPos) = [];

yp = NetWork(TestX,mf,AMR,Ac);
mre = MRE(yp, TestY)
Plot_Test(yp, TestY, DataSetName, mre, Union(AMR), length(unique(UsedPos)));