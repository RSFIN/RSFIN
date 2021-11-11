function [modelTwo,mre,N] = TwoLayerRSFIN(X,Y,mf,CluRe,k1,p1,k2,p2,wall,ExpMRE0,ExpMRE1,ExpMRE2)

load('..\user_data\Configuration2.mat');
n = size(X,1);
N = (k1+k2+p1+p2)*n;

Yd = Y;
Yd(Y<wall) = 50;
Yd(Y>=wall) = 100;
model_Classifier = [];
while isempty(model_Classifier)
    try
        model_Classifier  = RSFIN(X,Yd,mf,CluRe,5,5,ExpMRE0);
    end
end
yp_Class = NetWork(X,model_Classifier);

X1 = X(:,yp_Class < 75);Y1 = Y(yp_Class < 75);
if ~isempty(X1)
    model_1 = [];
    while isempty(model_1)
        try
            model_1 = RSFIN(X1,Y1,mf,CluRe,k1,p1,ExpMRE1);
        end
    end
else
    model_1 = [];
end
X2 = X(:,yp_Class >= 75);Y2 = Y(yp_Class >= 75);
if ~isempty(X2)
    model_2 = [];
    model_2 = RSFIN(X2,Y2,mf,CluRe,k2,p2,ExpMRE2);
    while isempty(model_2)
        try
            model_2 = RSFIN(X2,Y2,mf,CluRe,k2,p2,ExpMRE2);
        end
    end
    
else
    model_2 = [];
end

modelTwo.model_Classifier = model_Classifier;
modelTwo.model_1 = model_1;
modelTwo.model_2 = model_2;

yp = TwoLayerNetWork(X,modelTwo);
mre = MRE(yp, Y);
end

function yp = TwoLayerNetWork(X,modelTwo)

yp_C = NetWork(X,modelTwo.model_Classifier);

yp = yp_C;
for i = 1:length(yp)
    if yp_C(i) < 75
        yp(i) = NetWork(X(:,i), modelTwo.model_1);
    else
        yp(i) = NetWork(X(:,i), modelTwo.model_2);
    end
end
end

% Used to calculate the MRE value of experimental results
function res = MRE(yp, TestY)

% yp: forecast result
% TestY: Real function value
res = sum(abs(TestY-yp)./abs(TestY))/length(TestY)*100;
end