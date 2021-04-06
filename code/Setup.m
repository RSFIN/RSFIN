function [X, Y, CluRe, DataSetName, mf] = Setup(num)

DataSetNameSet = {'x264','SQL','sac','LLVM','javagc','hsmgp','hipacc','Dune','BDBJ','BDBC','Apache'};
DataSetName = DataSetNameSet{num};
XY = csvread(['..\data\Data\',DataSetName,'_AllNumeric.csv'],1);

X = XY(:,1:end-1)';
Y = XY(:,end)';
% Remove single variable of performance value
b = [];
for i = 1:size(X,1)
    if length(unique(X(i,:))) == 1
        b = [b,i];
    end
end
X(b,:) = [];

% Mean shift clustering to determine sample center.
CluRe = MeanShift(X);

% Determine the initial membership division.
mf = X2mf(X, Y, CluRe);
end

function mf = X2mf(X, Y, CluRe)

len = size(X,1);
CC = [];
for i = 1:len
    C = cov(X(i,:),Y);
    CC = [CC, C(2)];
end

for i = 1:len
    for j = 1:length(CluRe{i})
        mf(i).mf(j).type = 'gaussmf';
        mf(i).mf(j).config = [sqrt((CluRe{i}(2) - CluRe{i}(1))/10)/abs(CC(i)), CluRe{i}(j)];
    end
end
end

function CluRe = MeanShift(X)

CluRe = cell(size(X,1),1);

for i = 1:size(X,1)
    CluRe{i} = MS(X(i,:)');
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
r = 1/49 * (maxX - minX);
CluRe = linspace(minX,maxX,50);
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
