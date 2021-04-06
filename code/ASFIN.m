function [QMR, M] = ASFIN(X,mf,CluRe,NNN,ValX,ValY,TrainX,TrainY)

QMR = [];

QMR = GenMR(QMR,mf,CluRe,ValX,NNN);
tmre = TrainAndTest(TrainX,TrainY,ValX,ValY,mf,QMR);

flag = 0;
M = zeros(1,100);
for asd = 1:100
    
    if flag >= 10
        break;
    end
    AMR = Update_MR([QMR, GenMR(QMR,mf,CluRe,TrainX,NNN)],X,mf);
    
    [mf,~] = ANFIS(TrainX,TrainY,mf,AMR,1);
    
    AMR = Update_MR(AMR,X,mf);
    
    mre = TrainAndTest(TrainX,TrainY,ValX,ValY,mf,AMR);
    M(asd) = mre;
    flag = flag + 1;
    if mre < tmre
        flag = 0;
        tmre = mre;
        QMR = AMR;
    end
    disp([num2str(asd),'/100, mre: ',num2str(tmre)]);
end
M(asd:end) = M(asd-1);
end

function AMR = Update_MR(AMR,X,mf)
socer = SocerOfMR(X(:,randi(size(X,2),100,1)),mf,AMR);

if ~isempty(AMR(:, socer > 0.9))
    AMR(:, socer < 0.9) = [];
end

AMR = Union(AMR);
end

function mre = TrainAndTest(TrainX,TrainY,ValX,ValY,mf,AMR)
[mf,Ac] = ANFIS(TrainX,TrainY,mf,AMR,1);

yp = NetWork(ValX,mf,AMR,Ac);
mre = MRE(yp, ValY);
end

function MR = Union(MR)

n = size(MR,1);
Val = exp(-MR')*exp(MR) - n;
[i,j] = find(Val == 0);
j(i-j >= 0) = [];
MR(:,j) = [];
end

function MR = GenMR(AMR,mf,CluRe,X,N)

if isempty(AMR)
    MR = [];
    q = randperm(min(size(X,2),N));
    for i = 1:N
        MR = [MR, X2MR(X(:,q(i)),mf)];
    end
    MR = Union(MR);
else
    pos = randperm(size(AMR,2));
    MR = AMR(:,pos(1));
    posi = randperm(size(MR,1));
    
    set = CluRe{posi(1)};
    set(set == MR(posi(1))) = [];
    newi = randperm(length(set));
    MR(posi(1)) = newi(1);
end
end

function [socer, S] = SocerOfMR(X,mf,MR)

[ss, O3] = O2(X,mf,MR);

socer = zeros(size(MR,2),1);
yeta = Yeta(mf, zeros(size(MR,1),1));
for i = 1:length(socer)
    socer(i) = max(O3(i,:));
end

S = length(find(max(ss,[],1) > yeta))/size(X,2);
end

function [O2, O3] = O2(X,mf,MR)

[F, Aux] = mFun(X,mf);
bMR = reMR(Aux,MR,size(F,1));

O1 = F;
O2 = exp(bMR'*log(O1+0.001));
O3 = O2./(ones(size(O2,1))*O2);

end

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

function [bMR] = reMR(Aux,MR,s)

[n,m] = size(MR);
bMR = zeros(s,m);

for i = 1:m
    for j = 1:n
        bMR(Aux(j)+MR(j,i),i) = 1;
    end
end

end

function yeta = Yeta(mf, MR)

X_y = zeros(size(MR,1),1);
for i = 1:length(X_y)
    X_y(i) = mf(i).mf(MR(i) + 1).config(1) + mf(i).mf(MR(i) + 1).config(2);
end
yeta = O2(X_y,mf,MR);

end

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

function [T,mf,AMR] = Cluster(X, CluRe, beta, m)


T = zeros(size(X,2),1);
[mf,~] = genMR0(X(:,1),1,CluRe,beta);
AMR = [];

X_temp = X(:,randi(size(X,2)));
[~, MR_temp] = genMR0(X_temp,1,CluRe,beta);
sss = sum(O2(X,mf,MR_temp),1);
[~, k] = min(sss);
Xp = X(:, k);

for i = 1 : m
    
    
    [~,MR] = genMR0(Xp,1,CluRe,beta);
    T(O2(X,mf,MR) > Yeta(mf, MR)) = i;
    mf = mf;
    AMR = [AMR, MR];
    socer = Reward(X, mf, AMR);
    
    sss = sum(O2(X,mf,AMR),1);
    [~, k] = min(sss);
    Xp = X(:, k);
    
end

end

function socer = Reward(X, mf, AMR)

T = zeros(1,size(X,2));
for i = 1 : size(AMR,2)
    yeta = Yeta(mf, AMR(:,i));
    T = T + (O2(X,mf,AMR(:,i)) > yeta);
end
socer = length(find(T == 1)) / (length(find(T > 1)) + length(find(T == 0)));
end