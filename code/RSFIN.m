function  [model,mre,N,M] = RSFIN(X,Y,mf,CluRe,k,p,ExpMRE)

% CluRe: Sample cluster center
% k: Number of samples in the validation set (n)
% p: Number of samples in the train set (n)r


NNN = 1;% Initial number of rules
if size(X,1) == 16 && size(X,2) == 2560
    NNN = 32;
end
mre = 10000; % Initialize mre

M = [0,100];
while mre>=ExpMRE %Eliminate bad experiments
    [n, N] = size(X);
    Testpos = 1:N;
    
    [Valpos, TT, Testpos] = Sample(Testpos,k*n,p*n);%Validation set sampling and training set sampling
    
    ValX = X(:,Valpos); ValY = Y(Valpos);
    TrainX = X(:,TT); TrainY = Y(TT);
    TestX = X(:,Testpos);TestY = Y(Testpos);% Test set
    
    %
    % Rule search
    [QMR, ~] = ASFIN(X,mf,CluRe,NNN,ValX,ValY,TrainX,TrainY);
    % Learn
    [mf,Ac,S] = ANFIS([TrainX, ValX],[TrainY,ValY],mf,QMR,10);
    % Validation
    yp = NetWork(TestX,mf,QMR,Ac);
    
    mre = MRE(yp, TestY);
    
    M = [M;[toc, mean((yp - TestY).^2)/2]];
end

S(:,1) = S(:,1) - S(1,1);
S(:,1) = S(:,1) + M(end,1);
M = [M;S];

N = N - length(Testpos);
model.mf = mf;
model.MR = QMR;
model.Ac = Ac;

end

function [QMR, M] = ASFIN(X,mf,CluRe,NNN,ValX,ValY,TrainX,TrainY)

% X: configuration set
% mf: premise parameters
% CluRe: configure component cluster centers
% NNN: initial number of rules
% ValX,ValY: validation sample set
% TrainX,TrainY: train sample set

% Initialize the rule matrix
QMR = [];
QMR = GenMR(QMR,mf,CluRe,ValX,NNN);
% Initialize the mre
tmre = TrainAndTest(TrainX,TrainY,ValX,ValY,mf,QMR);

% Convergence mark
flag = 0;
M = zeros(1,100);
for asd = 1:100
    % Convergence condition
    if flag >= 10
        break;
    end
    
    % Eliminate rules that contribute less
    % AMR: Rule matrix for testing
    AMR = Update_MR([QMR, GenMR(QMR,mf,CluRe,ValX,NNN)],X,mf);
    % Modify premise parameters
    [mf,~] = ANFIS(TrainX,TrainY,mf,AMR,1);
    % Eliminate rules that contribute less
    AMR = Update_MR(AMR,X,mf);
    
    % Test network performance
    mre = TrainAndTest(TrainX,TrainY,ValX,ValY,mf,AMR);
    M(asd) = mre;
    
    flag = flag + 1;
    if mre < tmre
        flag = 0;
        tmre = mre;
        QMR = AMR;
    end
end
% Aligned output
M(asd:end) = M(asd-1);
end

function [mf,Ac,M] = ANFIS(X,Y,mf,MR,epoch)
% X: n*1
% F: s*1
% Aux: n*1
% MR: n*m
% bMR: s*m
% Ac: (n+1)*m

vdw = 0;
sdw = 0;
S = zeros(1,100);
T = zeros(1,100);

for t = 1:epoch
    [F, Aux] = mFun(X,mf);
    bMR = reMR(Aux,MR,size(F,1));
    
    O1 = F;
    O2 = exp(bMR'*log(O1+0.001));
    O3 = O2./(ones(size(O2,1))*O2);
    % Estimation of consequent parameters
    Ac = Structure_after(X,Y,O3);
    O4 = (Ac'*[X;ones(1,size(X,2))]).*O3;
    O5 = sum(O4,1);
    yp = O5;
    
    % Convergence condition
    S(t) = mean((Y-yp).^2)/2;
    T(t) = toc;
    if S(t) < 0.001
        break;
    end
    % Premise parameter update
    [mf,vdw,sdw] = reMF(X,PD(X,F,Ac,bMR),yp-Y,mf,vdw,sdw,t);
end
S(t:end) = S(t);
T(t:end) = T(t);
M = [T',S'];
end

function [yp] = NetWork(X,mf,MR,Ac)

% Use network parameters for calculations.
% X: n*1
% F: s*1
% Aux: n*1
% MR: n*m
% bMR: s*m
% Ac: (n+1)*m

% Parameter preparation before calculation
[F, Aux] = mFun(X,mf);
bMR = reMR(Aux,MR,size(F,1));

% ANFIS five-layer operation
O1 = F;
O2 = exp(bMR'*log(O1+0.0001));
O3 = O2./(ones(size(O2,1))*O2);
O4 = (Ac'*[X;ones(1,size(X,2))]).*O3;
O5 = sum(O4,1);

yp = O5;
end

function [Ac] = Structure_after(X,Y,O3)

m = size(O3,1);
[n,N] = size(X);
x = [];y = [];
for k = 1:N
    xt = [];
    for i = 1:m
        xt = [xt,[X(:,k)',1] * O3(i,k)];      
    end
    x = [x;xt];
    y = [y;Y(k)];
end
% Least squares estimation
Ac = pinv(x'*x)*x'*y;
Ac = reshape(Ac, n+1, m);

end

% Forward gradient calculation
function dYdF = PD(X,F,Ac,bMR)
%dYdF: s*N
F = F + 0.001;
D1 = (bMR*((Ac'*[X;ones(1,size(X,2))]).*exp(bMR'*log(F))))./F;
D2 = (1-bMR)*((Ac'*[X;ones(1,size(X,2))]).*exp(bMR'*log(F)));
D3 = (bMR*exp(bMR'*log(F)))./F;
D4 = (1-bMR)*exp(bMR'*log(F));

q = D2.*D3./(D1+0.0001) - D4;
a = D3;
b = D4;
dYdF = (q.*a)./(q.*F+b+0.001).^2;

end

function [mf,vdw,sdw] = reMF(X,dYdF,error,mf,vdw,sdw,t)
%The function of adaptive derivation with different membership degrees is 
% not implemented here (currently only gaussmf is processed)
s = 0;
beta1 = 0.9;
beta2 = 0.9;
alpha = 0.001;
epsilon = 0.001;
for i = 1:length(mf)
    
    x = X(i,:);
    for j = 1:length(mf(i).mf)
        s = s + 1;
        %%%
        sig = mf(i).mf(j).config(1);
        c = mf(i).mf(j).config(2);
        dFdc = exp(-(x-c).^2/2/sig^2).*(x-c)/sig^2;
        dFdsig = exp(-(x-c).^2/2/sig^2).*(x-c).^2/sig^3;
        
        dc(s) = error*(dYdF(s,:).*dFdc)';
        dsig(s) = error*(dYdF(s,:).*dFdsig)';
        %%%
    end
end

dw = [dc,dsig];%%%
%Adam
vdw = beta1*vdw + (1-beta1)*dw;
sdw = beta2*sdw + (1-beta2)*dw.^2;
vdwc = vdw/(1-beta1^t);
sdwc = sdw/(1-beta2^t);

delta = alpha * vdwc./(sqrt(sdwc)+epsilon);
dc = delta(1:length(dc));%%%
dsig = delta((length(dc)+1):end);%%%

s = 0;
for i = 1:length(mf)
    for j = 1:length(mf(i).mf)
        s = s + 1;
        mf(i).mf(j).config(1) = mf(i).mf(j).config(1) - dsig(s);%%%
        mf(i).mf(j).config(2) = mf(i).mf(j).config(2) - dc(s);%%%
    end
end
end

% Eliminate rules that contribute less
function AMR = Update_MR(AMR,X,mf)
socer = SocerOfMR(X(:,randi(size(X,2),100,1)),mf,AMR);

if ~isempty(AMR(:, socer > 0.9))
    AMR(:, socer < 0.9) = [];
end

AMR = Union(AMR);
end

% Network performance
function mre = TrainAndTest(TrainX,TrainY,ValX,ValY,mf,AMR)
[mf,Ac] = ANFIS(TrainX,TrainY,mf,AMR,1);

yp = NetWork(ValX,mf,AMR,Ac);
mre = MRE(yp, ValY);
end

% Eliminate the same rules
function MR = Union(MR)

n = size(MR,1);
Val = exp(-MR')*exp(MR) - n;
[i,j] = find(Val == 0);
j(i-j >= 0) = [];
MR(:,j) = [];
end

% Generate rules based on genetic algorithm
function MR = GenMR(AMR,mf,CluRe,ValX,N)

if isempty(AMR)
    MR = [];
    q = randperm(min(size(ValX,2),N));
    for i = 1:N
        MR = [MR, X2MR(ValX(:,q(i)),mf)];
    end
    MR = Union(MR);
else
    QMR = [];
    for i = 1:1
        pos = randperm(size(AMR,2));
        MR = AMR(:,pos(1));
        posi = randperm(size(MR,1));
        
        set = CluRe{posi(1)};
        set(set == MR(posi(1))) = [];
        newi = randperm(length(set));
        MR(posi(1)) = newi(1);
        QMR = [QMR, MR];
    end
    MR = QMR;
end
end

% Calculate the rule contribution value
function socer = SocerOfMR(X,mf,MR)

[~, O3] = O2(X,mf,MR);

socer = zeros(size(MR,2),1);

for i = 1:length(socer)
    socer(i) = max(O3(i,:));
end
end

% ANFIS second layer return value
function [O2, O3] = O2(X,mf,MR)

[F, Aux] = mFun(X,mf);
bMR = reMR(Aux,MR,size(F,1));

O1 = F;
O2 = exp(bMR'*log(O1+0.001));
O3 = O2./(ones(size(O2,1))*O2);

end

% ANFIS-Membership function value calculation
function [FA, Aux] = mFun(X,mf)
FA = [];
for x_n = 1:size(X,2)
    F = [];Aux = [];
    for i = 1:length(mf)
        Aux = [Aux; length(F)+1];
        for j = 1:length(mf(i).mf)
            F = [F; MF(mf(i).mf(j).type,X(i,x_n),mf(i).mf(j).config)];
        end
    end
    FA = [FA,F];
end
end

% ANFIS-rule matrix conversion
function [bMR] = reMR(Aux,MR,s)

[n,m] = size(MR);
bMR = zeros(s,m);

for i = 1:m
    for j = 1:n
        bMR(Aux(j)+MR(j,i),i) = 1;
    end
end

end

% Convert data into rules
function [MR] = X2MR(X, mf)

MR = zeros(size(X));
for i = 1:length(X)
    m = zeros(length(mf(i).mf),1);
    for j = 1:length(mf(i).mf)
        m(j) = MF(mf(i).mf(j).type,X(i),mf(i).mf(j).config);
    end
    [~,MR(i)] = max(m);
end
MR = MR - 1;
end

% Used to calculate the MRE value of experimental results
function res = MRE(yp, TestY)

% yp: forecast result
% TestY: Real function value
res = sum(abs(TestY-yp)./abs(TestY))/length(TestY)*100;
end