clc,clear,clf
[X, Y, CluRe, DataSetName, mf] = Setup(1); % {'x264','SQL','sac','LLVM','javagc','hsmgp','hipacc','Dune','BDBJ','BDBC','Apache'}
% pos = randi(size(X,2),1,200);
% X = X(:,pos);
% Y = Y(pos);
UsedPos = [];
[n, N] = size(X);

ValPos = randi(N, 1, 2*n);
% ValPos = 1:5*n;
% ValPos = 1:size(X,2);
% UsedPos = [UsedPos, ValPos];

%Iteration for train
AMR = [];
AAc = [];
TT = [];
MR = ones(n,1);
Ac = ones(n+1,1);
yp = NetWork(X(:,ValPos),mf,MR,Ac);
ValY = Y(ValPos);
v_MRE = abs(ValY-yp)./ValY*100;

epoch = 0;
while epoch<1000
    epoch = epoch + 1;
% for i = 1:1
% ValData

% temp_pos = ValPos(v_MRE == max(v_MRE));
% temp_pos = temp_pos(1);
% temp_pos = ValPos(v_MRE>1);
% temp_pos = temp_pos(randi(length(temp_pos)));
temp_pos = ValPos(randi(length(ValPos)));

% GenRule
MR = X2MR(X(:,temp_pos), mf);
% mutation_pos = randi(length(MR));
% MR(mutation_pos) = randi(length(CluRe{mutation_pos})) - 1;
% TrainData

% yeta = Yeta(mf, MR);
% TrainPos = find(O2(X,mf,MR) > 0.01*yeta);
% TrainPos = TrainPos(randi(length(TrainPos),1,3));%%%%
[~,TrainPos] = sort(O2(X,mf,MR),'descend');
TrainPos = TrainPos(1:3);

% TrainPos = [TrainPos, ValPos];

TrainPos(TrainPos == temp_pos) = [];
UsedPos = [UsedPos, TrainPos];
TT = unique([TT,TrainPos]);
[~,Pos] = sort(O2(X(:,TT),mf,MR),'descend');
TrainPos = TT(Pos(1:min([n,length(TT)])));

TrainX = X(:,TT); TrainY = Y(TT);
[mf,Ac] = ANFIS(TrainX,TrainY,mf,MR,1);

% yp = NetWork(X(:,ValPos),mf,[AMR,MR],[AAc,Ac]);
% ValY = Y(ValPos);
% v_MRE = abs(ValY-yp)./ValY*100
% if length(find(v_MRE < 5) )>= 1
%     ValPos(v_MRE < 5) = [];
    AMR = [AMR,MR];
    AAc = [AAc,Ac];
% end
socer = SocerOfMR(X,mf,AMR);
AMR(:,socer<0.8) = [];
AAc(:,socer<0.8) = [];


TestX = X;TestY = Y;
TestX(:,UsedPos) = [];
TestY(UsedPos) = [];
% TestX = X(:,ValPos);
% TestY = ValY;

yp = NetWork(TestX,mf,AMR,AAc);
mre = MRE(yp, TestY)
% Plot_Test(yp, TestY, DataSetName, mre, Union(AMR), length(unique(UsedPos)));
mm(epoch) = mre;
if epoch > 10
    if std(mm(end-9:end)) < 0.05
        break;
    end
end
end
plot(mm)
Plot_Test(yp, TestY, DataSetName, mre, Union(AMR), length(unique(UsedPos)));
