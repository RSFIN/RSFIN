QQ = MMM;
QQ(:,22:100) = [];
QQ(QQ>40) = nan;
QQ(:,40:end) = [];
for i = 1:size(QQ,2)
    pos = find(isnan(QQ(:,i)));
    QQ(pos,i) = 0; 
    for j = 1:100
    QQ(pos,i) = mean(QQ(:,i)); 
    end
end
hold on 
plot(QQ');

