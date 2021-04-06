function [mf,MR] = genMR0(X,m)

% Xp = randi(2,size(X,1),50 - size(X,2))-1;
% X = [X,Xp];

[n,N] = size(X);%数据维度
%初始规则库数量
T = HC(X,m)';
%隶属度函数上限
MM = 2*ones(n,1);
%初始化隶属度函数

%根据规则库聚类选择合适的隶属度函数
for R_n = 1:m
    c(R_n,:) = mean(X(:,T==R_n),2)';
    sig(R_n,:) = std(X(:,T==R_n),[],2)'+0.1;
end

KL = @(c1,sig1,c2,sig2) -1/2*(log(sig1^2/sig2^2)-sig1^2/sig2^2-(c1-c2)^2/sig2^2+1);

for i = 1:n
    s = 0;Temp = [1,2];
    for a1 = 1:m-1
        for a2 = a1+1:m
            distance = KL(c(a1,i),sig(a1,i),c(a2,i),sig(a2,i));
            if distance > s
                s = distance;
                Temp(1) = a1;
                Temp(2) = a2;
            end
        end
    end
    mf(i).mf(1).config = [sig(Temp(1),i)+0.1,c(Temp(1),i)];
    mf(i).mf(1).type = 'gaussmf';
    mf(i).mf(2).config = [sig(Temp(2),i)+0.1,c(Temp(2),i)];
    mf(i).mf(2).type = 'gaussmf';
end


MR = zeros(n,m);
for i = 1:n
    for j = 1:m
        if KL(mf(i).mf(1).config(2),mf(i).mf(1).config(1),c(j,i),sig(j,i)) > ...
                KL(mf(i).mf(2).config(2),mf(i).mf(2).config(1),c(j,i),sig(j,i))
            MR(i,j) = 1;
        else
            MR(i,j) = 0;
        end
    end
end

MR = Union(MR);