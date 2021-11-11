% for x264
clf

Nx = csvread('plot_x264.csv');
pos = randi(size(Nx,1),30,1);
Nx = Nx(pos,:);
plot3(Nx(:,1),Nx(:,2),Y(pos),'o');
xlabel('Reduced component 1');
ylabel('Reduced component 2');
zlabel('Performance');
grid on

%%
[~,label] = max(O2(X(:,pos),mf,MR));
pos1 = find(label == 1);
pos2 = find(label == 2);
hold on
plot3(Nx(pos1,1),Nx(pos1,2),Y(pos(pos1)),'ro');
plot3(Nx(pos2,1),Nx(pos2,2),Y(pos(pos2)),'bo');
xlabel('Reduced component 1');
ylabel('Reduced component 2');
zlabel('Performance');
grid on
%%
x = Nx(pos2,1);
y = Nx(pos2,2);
z = 265+0.2662*x+0.5813*y;
plot3(x,y,z,'o');
xlabel('Reduced component 1');
ylabel('Reduced component 2');
zlabel('Performance');
x12 = Nx(pos2,1);
x22 = Nx(pos2,2);
y2 = Y(pos2);
x = -40:2:40;
y = -40:2:40;
colormap(spring)
[xx,yy] = meshgrid(x,y,'b');
zz = 265+0.2662*xx+0.5813*yy;
hold on 
mesh(xx,yy,zz);
%%
x = Nx(pos1,1);
y = Nx(pos1,2);
z = 642.5+3.669*x+7.961*y;
plot3(x,y,z,'o');
xlabel('Reduced component 1');
ylabel('Reduced component 2');
zlabel('Performance');
% x11 = Nx(pos1,1);
% x21 = Nx(pos1,2);
% y1 = Y(pos1);
% x = -40:2:40;
% y = -40:2:40;
% colormap(bone)
% [xx,yy] = meshgrid(x,y);
% zz = 642.5+3.669*xx+7.961*yy;
% hold on 
% mesh(xx,yy,zz);