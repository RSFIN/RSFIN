function [mf,Ac,mre] = ANFIS(X,Y,mf,MR,epoch)
% X: n*1
% F: s*1
% Aux: n*1
% MR: n*m
% bMR: s*m
% Ac: (n+1)*m

vdw = 0;
sdw = 0;
mre = zeros(1,100);
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
    mre(t) = sum(abs(Y-yp)./Y)/length(Y)*100;
    if mre(t) < 0.001
        break;
    end
    % Premise parameter update
    [mf,vdw,sdw] = reMF(X,PD(X,F,Ac,bMR),yp-Y,mf,vdw,sdw,t);
end
mre(t:end) = mre(t);
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
