
clear; close all;
k = 2;
n = 1000;
[X,t] = kmeansRnd(2,k,n);

[x1,x2] = meshgrid(linspace(min(X(1,:)),max(X(1,:)),n), linspace(min(X(2,:)),max(X(2,:)),n));
[model, llh] = logitReg(X,t-1,0);
[y,p] = logitPred(model,X);

w = model.w;
w0 = model.w0;
plot(llh);
figure;
spread(X,t);

y = w(1)*x1+w(2)*x2+w0;

hold on;
contour(x1,x2,y,1);
hold off;
%%
% clear; close all;
% k = 3;
% n = 200;
% [X,t] = rndKCluster(2,k,n);
% 
% [x1,x2] = meshgrid(linspace(min(X(1,:)),max(X(1,:)),n), linspace(min(X(2,:)),max(X(2,:)),n));
% [model, llh] = mnReg(X,t, 1e-4,2);
% plot(llh);
% figure;
% spread(X,t);
% 
% W = model.W;
% % y = w(1)*x1+w(2)*x2+w(3);
% 
% hold on;
% contour(x1,x2,t,1);
% hold off;