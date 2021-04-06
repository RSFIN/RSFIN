clc,clear
%初始化
DataSetNameSet = {'x264','SQL','sac','LLVM','javagc','hsmgp','hipacc','Dune','BDBJ','BDBC','Apache'};
Socer = cell(11,6);
% load '..\user_data\Socer'
for q = 6:6
    DataSetName = DataSetNameSet{q};
    XY = csvread(['..\data\Data\',DataSetName,'_AllNumeric.csv'],1);
    Socer{q,1} = DataSetName;
    
    n = (size(XY,2) - 1);
    for k = 3:5
        DataSize = n * k;
        
        dataPos = randi(size(XY,1), DataSize, 1);
        CluRe = MeanShift(XY(:,1:end-1)');
        sss = [];
        for j = 1:30
            for i = 1:5
                dataPos = randi(size(XY,1),1,1);
                TestX = XY(dataPos,1:end-1)';
                TestY(i) = XY(dataPos,end)';
                m = 2;
                epoch = 20;
                
                [mf,MR] = genMR0(TestX,m,CluRe,0.01);
                
                dis = O2(XY(:,1:end-1)',mf,MR);
                [a,b] = sort(dis,'descend');
                dataPos = b(2:DataSize + 1);
                X = XY(dataPos,1:end-1)';
                Y = XY(dataPos,end)';
                
                [mf,Ac] = ANFIS(X,Y,mf,MR,10);
                yp(i) = NetWork(TestX,mf,MR,Ac);               
            end
            MRE = sum(abs(TestY-yp)./TestY)/length(TestY)*100;
            sss = [sss, MRE];
            Socer{q,k+1} = sss;
            save('..\user_data\Socer','Socer');
            disp([DataSetName, '   k = ', num2str(k), '   ', num2str(j/30*100), '%']);
        end
    end
end
%%

clf
hold on
plot(TestY);
% plot(TestYp);
plot(yp);
% legend('真实数据',['SPL Conqueror  MRE:',num2str(MREp)],...
%           ['Oures                 MRE:',num2str(MRE)]);
legend('真实数据', ['Oures                 MRE:',num2str(MRE)]);
title(['数据集：',DataSetName,...
    '  规则数：',num2str(size(MR,2)),...
    '  样本数：',num2str(DataSize)]);