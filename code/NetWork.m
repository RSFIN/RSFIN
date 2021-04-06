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
