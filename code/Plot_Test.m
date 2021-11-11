function [] = Plot_Test(yp, TestY, DataSetName, MRE, MR, DataSize)

% yp: forecast result
% TestY: Real function value
% DataSetName
% MRE
% MR: Matrix of Rule 
% DataSize

clf
hold on
plot(TestY);
plot(yp);

legend('Real data', ['Oures                 MRE:',num2str(MRE)]);
title(['Data set£º',DataSetName,...
    '  Number of rules£º',num2str(size(MR,2)),...
    '  Number of running samples£º',num2str(DataSize)]);