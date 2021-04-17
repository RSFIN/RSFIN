DataSetNameSet = {'x264','SQL','sac','LLVM','javagc','hsmgp','hipacc','Dune','BDBJ','BDBC','Apache'};
Deep = csvread('box.csv',0,1);


SCORE = csvread(['..\user_data\result_',DataSetNameSet{1},'.csv']);
DDD = SCORE(:,2);
DDD = [DDD,Deep(1,:)'];
SCORE = csvread(['..\user_data\result_',DataSetNameSet{2},'.csv']);
DDD = [DDD,SCORE(:,2)];
DDD = [DDD,Deep(2,:)'];
SCORE = csvread(['..\user_data\result_',DataSetNameSet{4},'.csv']);
DDD = [DDD,SCORE(:,2)];
DDD = [DDD,Deep(3,:)'];
SCORE = csvread(['..\user_data\result_',DataSetNameSet{6},'.csv']);
DDD = [DDD,SCORE(:,2)];
DDD = [DDD,Deep(8,:)'];
SCORE = csvread(['..\user_data\result_',DataSetNameSet{7},'.csv']);
DDD = [DDD,SCORE(:,2)];
DDD = [DDD,Deep(7,:)'];
SCORE = csvread(['..\user_data\result_',DataSetNameSet{8},'.csv']);
DDD = [DDD,SCORE(:,2)];
DDD = [DDD,Deep(9,:)'];
SCORE = csvread(['..\user_data\result_',DataSetNameSet{9},'.csv']);
DDD = [DDD,SCORE(:,2)];
DDD = [DDD,Deep(6,:)'];
SCORE = csvread(['..\user_data\result_',DataSetNameSet{10},'.csv']);
DDD = [DDD,SCORE(:,2)];
DDD = [DDD,Deep(5,:)'];
SCORE = csvread(['..\user_data\result_',DataSetNameSet{11},'.csv']);
DDD = [DDD,SCORE(:,2)];
DDD = [DDD,Deep(4,:)'];
clf
h=notBoxPlot(DDD,'jitter',0.65);
set(gca,'XTick',1.5:2:17.5);
set(gca,'XTicklabel',{'x264','SQL','LLVM','HSMGP','HIPAcc','Dune MGS','BDB-J','DBD-C','Apache'});
d=[h.data];
sd = [h.sdPtch];
se = [h.semPtch];
mu = [h.mu];
set(sd(2:end),'handlevisibility','off')
set(se(2:end),'handlevisibility','off')
set(mu(2:end),'handlevisibility','off')
set(d(2:6),'handlevisibility','off')
legend('Quartiles','95% confidence intervals \n for the mean','Mean','RSFIN','RSFIN + Classifier','DeepPerf');
set(d(2:2:end),'markerfacecolor',[0.4,1,0.4],'color',[0,0.4,0])
set(d([7,13,15]),'markerfacecolor',[1,0.4,0.4],'color',[0.4,0,0])