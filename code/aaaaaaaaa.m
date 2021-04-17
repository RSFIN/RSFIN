Nx = csvread('plot7_1000.csv');

[~,label] = max(O2(X,mf,QMR));
pos1 = find(label == 1);
pos2 = find(label == 2);
hold on
yp = NetWork(X,mf,QMR,Ac);
plot3(Nx(pos1,1),Nx(pos1,2),Y(pos1),'ro');
plot3(Nx(pos2,1),Nx(pos2,2),Y(pos2),'bo');
plot3(Nx(pos1,1),Nx(pos1,2),yp(pos1),'go');
plot3(Nx(pos2,1),Nx(pos2,2),yp(pos2),'yo');
legend('a_1','a_2','pre(a_1)','pre(a_2)')