function [] = Transfor(type,Config)

warning off

x = (0:0.01:1)';
for i=1:length(x)
    y(i) = MF(type,x(i),Config);
end
y = y';
Dic = [{'gaussmf'},{'gbellmf'},{'trapmf'},{'sigmf'},{'trimf'}];

AConfig = cell(length(Dic),2);

for i = 1:length(Dic)
    strfun = MF(Dic{i},[],[],0);
    f=fittype(strfun{1},'independent','x','coefficients',strfun{2});
    t = linspace(0.4,0.6,length(strfun{2}));
    cfun=fit(x,y,f,'Start',t)
    
end

