function [mf,MR,Ac] = GAforMR(X,Y,mf,MR,epoch)
[mf,Ac] = ANFIS(X,Y,mf,MR,3);
yp = NetWork(X,mf,MR,Ac);
MRscoer = sum((Y-yp).^2)/length(Y);
for epochs = 1:epoch
    %交叉产生新规则
    NewMR = MR;
    
    for i = 1:size(MR,2)
        temp = randi(size(MR,1));
        NewMR(temp,i) = 1 - NewMR(temp,i);
    end
    
    scoer = zeros(1,size(NewMR,2));
    for i = 1:length(scoer)
        [mf,Ac] = ANFIS(X,Y,mf,[MR,NewMR(:,i)],3);
        yp = NetWork(X,mf,[MR,NewMR(:,i)],Ac);
        scoer(i) = sum((Y-yp).^2)/length(Y);
    end
    [~,s] = min(scoer);
    BestNewMR = NewMR(:,s);
    if min(scoer) < MRscoer || rand()>0.9
        MRscoer = min(scoer);
        MR = [MR,BestNewMR];
        [mf,Ac] = ANFIS(X,Y,mf,MR,3);
    end
    
    %淘汰旧规则
    scoer = zeros(1,size(MR,2)-1);
    for i = 1:length(scoer)
        NewMR = MR;
        NewMR(:,i) = [];
        [mf,Ac] = ANFIS(X,Y,mf,NewMR,3);
        yp = NetWork(X,mf,NewMR,Ac);
        scoer(i) = sum((Y-yp).^2)/length(Y);
    end
    if min(scoer) < MRscoer && rand()<0.95
        [MRscoer,s] = min(scoer);
        MR(:,s) = [];
        [mf,Ac] = ANFIS(X,Y,mf,MR,3);
    end
    [mf,Ac] = ANFIS(X,Y,mf,MR,3);
    if MRscoer < 20000
        break;
    end
end
