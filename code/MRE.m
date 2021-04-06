function res = MRE(yp, TestY)

res = sum(abs(TestY-yp)./TestY)/length(TestY)*100;