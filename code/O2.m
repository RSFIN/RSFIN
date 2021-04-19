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