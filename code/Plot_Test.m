function [] = Plot_Test(yp, TestY, DataSetName, MRE, MR, DataSize)

clf
hold on
plot(TestY);
plot(yp);

legend('真实数据', ['Oures                 MRE:',num2str(MRE)]);
title(['数据集：',DataSetName,...
    '  规则数：',num2str(size(MR,2)),...
    '  样本数：',num2str(DataSize)]);