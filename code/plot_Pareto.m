
hold on
plot(A(:,1),A(:,2),'ro','MarkerSize',15);
plot(B(:,1),B(:,2),'ms','MarkerSize',15);
plot(C(:,1),C(:,2),'bp','MarkerSize',15);
plot(D(:,1),D(:,2),'kd','MarkerSize',15);
DataSetNameSet = {'x264','SQL','LLVM','Apache','BDBC','BDBJ','hipacc','hsmgp','Dune'};
Deep = csvread('box.csv',0,1);
DeepTime = [145,219,141,146,154,170,419,176,162];
DeepMagin = [0.28,0.14,0.18,0.48,0.91,0.06,0.77,0.18,0.93];
TimeMagin = [23,15,16,25,17,5,23,11,19];
A = [];
B = [];
C = [];
D = [];
for N_Data =1:9
    DataSetName = DataSetNameSet{N_Data};
    
    SCORE2 = csvread(['..\user_data\result_',DataSetName,'.csv']);
    SCORE3 = csvread(['..\user_data\result_TwoLayer_',DataSetName,'.csv']);
    
    A = [A;[mean(Deep(N_Data,:)),DeepTime(N_Data)]];
    plot([mean(Deep(N_Data,:))-DeepMagin(N_Data),mean(Deep(N_Data,:))+DeepMagin(N_Data)],[DeepTime(N_Data),DeepTime(N_Data)],'r');
    plot([mean(Deep(N_Data,:)),mean(Deep(N_Data,:))],[DeepTime(N_Data)-TimeMagin(N_Data),DeepTime(N_Data)+TimeMagin(N_Data)],'r');
    if N_Data <=6
        DECART = csvread(['..\user_data\CART_',DataSetName,'_Datails.csv'],91,5);
        B = [B;[mean(DECART(:,1)),mean(DECART(:,2))]];
        plot([mean(DECART(:,1))-1.96*std(DECART(:,1))/sqrt(SCORE2(1, 3)),mean(DECART(:,1))+1.96*std(DECART(:,1))/sqrt(SCORE2(1, 3))],[mean(DECART(:,2)),mean(DECART(:,2))],'m');
        plot([mean(DECART(:,1)),mean(DECART(:,1))],[mean(DECART(:,2))-1.96*std(DECART(:,2))/sqrt(SCORE2(1, 3)),mean(DECART(:,2))+1.96*std(DECART(:,2))/sqrt(SCORE2(1, 3))],'m');
    end
    if N_Data ~= 5
        C = [C;[mean(SCORE2(:,2)),mean(SCORE2(:,5))]];
        plot([mean(SCORE2(:,2))-1.96*std(SCORE2(:,2))/sqrt(SCORE2(1, 3)),mean(SCORE2(:,2))+1.96*std(SCORE2(:,2))/sqrt(SCORE2(1, 3))],[mean(SCORE2(:,5)),mean(SCORE2(:,5))],'b');
        plot([mean(SCORE2(:,2)),mean(SCORE2(:,2))],[mean(SCORE2(:,5))-1.96*std(SCORE2(:,5))/sqrt(SCORE2(1, 3)),mean(SCORE2(:,5))+1.96*std(SCORE2(:,5))/sqrt(SCORE2(1, 3))],'b');
    end
    if N_Data > 3
        D = [D;[mean(SCORE3(:,2)),mean(SCORE3(:,5))]];
        plot([mean(SCORE3(:,2))-1.96*std(SCORE3(:,2))/sqrt(SCORE2(1, 3)),mean(SCORE3(:,2))+1.96*std(SCORE3(:,2))/sqrt(SCORE2(1, 3))],[mean(SCORE3(:,5)),mean(SCORE3(:,5))],'k');
        plot([mean(SCORE3(:,2)),mean(SCORE3(:,2))],[mean(SCORE3(:,5))-1.96*std(SCORE3(:,5))/sqrt(SCORE3(1, 3)),mean(SCORE3(:,5))+1.96*std(SCORE3(:,5))/sqrt(SCORE2(1, 3))],'k');
    end
end
a = mean(A);b = mean(B);c = mean(C);d = mean(D);


% plot([0,a(1)],[0,a(2)],'*:r','MarkerSize',8);
% plot([0,b(1)],[0,b(2)],'*:m','MarkerSize',8);
% plot([0,c(1)],[0,c(2)],'*:b','MarkerSize',8);
% plot([0,d(1)],[0,d(2)],'*:k','MarkerSize',8);
legend('DeepPerf','DECART','RSFIN','Two-Layer RSFIN');
xlabel('MRE');
ylabel('Time Cost');