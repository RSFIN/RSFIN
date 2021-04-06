function [mf,Ac,MR] = NAS_FIS(X,Y,m0,ValX,ValY,Epoch)

epoch1 = 20;
% epoch2 = 10;
epoch3 = 10;
[mf,MR] = genMR0(X,m0);
for i = 1:Epoch
    [mf,Ac] = ANFIS(X,Y,mf,MR,epoch1);
    % [mf,Ac] = NASforMF(X,Y,mf,MR,epoch2);
    [mf,MR,Ac] = GAforMR(ValX,ValY,mf,MR,epoch3);
end

[mf,Ac] = ANFIS(X,Y,mf,MR,epoch1);