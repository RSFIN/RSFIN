function information_print(model, DataSetName)

XY = importdata(['..\data\Data\',DataSetName,'_AllNumeric.csv']);
options = XY.textdata;
CluRe = MS(XY.data(:,end));
PREF_tag = Perf_tag(length(CluRe));

for i = 1:size(model.MR,2)
    fprintf(['Rule ',  num2str(i),':\n']);
    fprintf('IF \n    ');
    for j = 1:length(model.mf)
        if j>=2
            fprintf('^\t');
        end
        fprintf('%s = %d\n', options{j}, round(model.mf(j).mf(model.MR(j,i)+1).config(2)));
    end
    fprintf('THEN \n    PREF ');
    X_test = zeros(length(model.mf),1);
    for j = 1:length(model.mf)
        X_test(j) = model.mf(j).mf(model.MR(j,i)+1).config(2);
    end
    Y_test = NetWork(X_test,model);
    [~, PREF_pos] = min(abs(CluRe - Y_test));
    fprintf(PREF_tag{PREF_pos});
    fprintf('\n');
end

end

function CluRe = MS(X)

m = 1;
if exp(max(X)) == inf
    m = max(X);
    X = X/max(X);
end
minX = min(X);
maxX = max(X);
r = 1/4 * (maxX - minX);
CluRe = linspace(minX,maxX,5);
pCluRe = CluRe;

while 1
    dis = abs(log(exp(-X)*exp(CluRe)));
    [a,b] = find(dis < r);
    label = unique(b);
    for i = 1:length(label)
        CluRe(label(i)) = mean(X(a(b == label(i))));
    end
    if max(abs(CluRe-pCluRe)) < 1e-10
        CluRe = unique(CluRe(label)) * m;
        return;
    end
    CluRe = unique(CluRe(label));
    pCluRe = CluRe;
end
end

function Tags = Perf_tag(LenOfClure)

switch LenOfClure
    case 1
        Tags = {'0'};
        return
    case 2
        Tags = {'-','+'};
        return
    case 3
        Tags = {'-','0','+'};
        return
    case 4
        Tags = {'--','-','+','++'};
        return
    case 5
        Tags = {'--','-','0','+','++'};
        return
    otherwise
        Tags = [];
        return
end
        

end
