function T = HC(X,k)

KL = @(c1,c2,s1,s2)(-1/2*sum(log2(power(s1+0.1,2))-log2(power(s2+0.1,2))-power(s1+0.1,2)...
    ./power(s2+0.1,2)-power(c1-c2,2)./power(s2+0.1,2)+1));

clu0 =[];
for i = 1:size(X,2)
    clu0(i).num = i;
    clu0(i).sub = i;
    clu0(i).mu = X(:,i);
    clu0(i).sig = 0.001*ones(size(X(:,i)));
end

for times = 1:(size(X,2)-k)
    
    mindis = inf;
    for i = 1:length(clu0)-1
        for j = i+1:length(clu0)
            dis = KL(clu0(i).mu,clu0(j).mu,clu0(i).sig,clu0(j).sig);
            if dis < mindis
                mindis = dis;
                a(1) = i;
                a(2) = j;
            end
        end
    end
    clu0(a(1)).sub = [clu0(a(1)).sub,clu0(a(2)).sub];
    clu0(a(1)).mu = mean(X(:,clu0(a(1)).sub),2);
    clu0(a(1)).sig = std(X(:,clu0(a(1)).sub),[],2);
    clu0(a(2)) = [];
end

T = zeros(size(X,2),1);
for i = 1:length(clu0)
    clu0(i).num = i;
    T(clu0(i).sub) = i;
end