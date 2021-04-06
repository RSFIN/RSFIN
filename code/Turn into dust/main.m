clc,clear
%Data
load('..\data\TestData.mat')
n_f = 1;
n_config = 4;
config = Data(:,1:n_config);
f = Data(:,end-n_f+1:end);

%Structure
%Structure_pre
for i = 1:n_config
    mufun(i).num = 2;
    for j = 1:mufun(i).num
        mufun(i).mf(j).type = 'gbellmf';
        mufun(i).mf(j).config = rand(1,3);
    end
end
%Gene code
gene = config;


%Network
%level 1
pos = 1;

for k = 1:size(config,1)
    for i = 1:n_config
        for j = 1:mufun(i).num
            L1(k,i,j) = MF(mufun(i).mf(j).type,config(k,i),mufun(i).mf(j).config);
        end
    end
end
%level 2
for k = 1:size(config,1)
    for i = 1:size(gene,1)
        L2(i,k) = 1;
        for j = 1:length(gene(i,:))
            L2(i,k) = L2(i,k) * L1(k,j,gene(i,j)+1);
        end
    end
end

%level 3
for k = 1:size(config,1)
    for i = 1:length(L2)
        L3(i,k) = L2(i,k)/sum(L2(:,k));
    end
end

%Structure_after
Ac = zeros(size(gene,1),size(gene,2)+1);
for i = 1:size(gene,1)
    x = [];y = [];
        for k = 1:size(L3,2)
            x = [x;[config(k,:),1]];
            y = [y;f(k).*L3(i,k)];
        end
    Ac(i,:) = pinv(x'*x)*x'*y;
end 

%level 4
for j = 1:size(L3,1)
    for k = 1:size(L3,2)
        L4(j,k) = L3(j,k)*[config(k,:),1]*Ac(j,:)';
    end
end

%level 5
for i = 1:size(L4,2)
    L5(i) = sum(L4(:,i));
end