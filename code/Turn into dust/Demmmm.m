% clc,clear;
DataSetName = 'Apache';%  Apache BerkeleyC BerkeleyJ LLVM sqlite x264
SamplMethod = 'FW';% FW HO HS PW
% XY = csvread(['..\data\ICSE2012\material\',DataSetName,...
%     '\',SamplMethod,'\accuracy.csv'],1);
XY = csvread(['..\data\Data\',DataSetName,'_AllNumeric.csv'],1);
DataSize = 8*5;
dataPos = randi(size(XY,1),DataSize,1);
X = XY(dataPos,1:end-1)';
Y = XY(dataPos,end)';
m0 = 8;
dataPos = randi(size(XY,1),20,1);
ValX = XY(dataPos,1:end-1)';
ValY = XY(dataPos,end)';
epoch = 3;
[mf,Ac,MR] = NAS_FIS(X,Y,m0,ValX,ValY,epoch);

% dataPos = randi(size(XY,1),20,1);
dataPos = randi(size(XY,1),20,1);
TestX = XY(dataPos,1:end-1)';
TestY = XY(dataPos,end)';
% TestYp = XY(dataPos,end)';
yp = NetWork(TestX,mf,MR,Ac);
MRE = sum(abs(TestY-yp)./TestY)/length(TestY)*100;
% MREp = sum(abs(TestY-TestYp)./TestY)/length(TestY)*100;
clf
hold on
plot(TestY);
% plot(TestYp);
plot(yp);
% legend('真实数据',['SPL Conqueror  MRE:',num2str(MREp)],...
%           ['Oures                 MRE:',num2str(MRE)]);
legend('真实数据', ['Oures                 MRE:',num2str(MRE)]);
title(['数据集：',DataSetName,'-',SamplMethod,...
    '  规则数：',num2str(size(MR,2)),...
    '  样本数：',num2str(DataSize)]);
%% 存储模型
save(['..\user_data\mf_',DataSetName,'_',SamplMethod],'mf');
save(['..\user_data\MR_',DataSetName,'_',SamplMethod],'MR');
save(['..\user_data\Ac_',DataSetName,'_',SamplMethod],'Ac');
save(['..\user_data\X_',DataSetName,'_',SamplMethod],'X');
save(['..\user_data\Y_',DataSetName,'_',SamplMethod],'Y');
%% 导入模型
load(['..\user_data\mf_',DataSetName,'_',SamplMethod]);
load(['..\user_data\MR_',DataSetName,'_',SamplMethod]);
load(['..\user_data\Ac_',DataSetName,'_',SamplMethod]);
load(['..\user_data\X_',DataSetName,'_',SamplMethod]);
load(['..\user_data\Y_',DataSetName,'_',SamplMethod]);