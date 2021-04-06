function res = MRE(yp, TestY)

% Used to calculate the MRE value of experimental results
% yp: forecast result
% TestY: Real function value
res = sum(abs(TestY-yp)./TestY)/length(TestY)*100;