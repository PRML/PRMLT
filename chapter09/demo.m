

%% demo: EM linear regression
% close all; clear;
% d = 5;
% n = 200;
% [x,t] = linRnd(d,n);
% [model,llh] = linRegEm(x,t);
% plot(llh);

%% classification
% clear; close all
% k = 2;
% d = 2;
% n = 1000;
% [X,t] = kmeansRnd(d,k,n);
% [x1,x2] = meshgrid(linspace(min(X(1,:)),max(X(1,:)),n), linspace(min(X(2,:)),max(X(2,:)),n));
% 
% [model, llh] = rvmBinEm(X,t-1);
% plot(llh);
% y = rvmBinPred(model,X)+1;
% figure;
% binPlot(model,X,y);
%% demo: kmeans 
% close all; clear;
% d = 2;
% k = 3;
% n = 500;
% [X,label] = kmeansRnd(d,k,n);
% y = kmeans(X,k);
% plotClass(X,label);
% figure;
% plotClass(X,y);

%% demo: Em for Gauss Mixture 
% close all; clear;
% d = 2;
% k = 3;
% n = 1000;
% [X,label] = mixGaussRnd(d,k,n);
% plotClass(X,label);
% 
% m = floor(n/2);
% X1 = X(:,1:m);
% X2 = X(:,(m+1):end);
% % train
% [z1,model,llh] = mixGaussEm(X1,k);
% figure;
% plot(llh);
% figure;
% plotClass(X1,z1);
% % predict
% z2 = mixGaussPred(X2,model);
% figure;
% plotClass(X2,z2);
%% demo: Em for Gauss mixture initialized with kmeans;
% close all; clear;
% d = 2;
% k = 3;
% n = 500;
% [X,label] = mixGaussRnd(d,k,n);
% init = kmeans(X,k);
% [z,model,llh] = mixGaussEm(X,init);
% plotClass(X,label);
% figure;
% plotClass(X,init);
% figure;
% plotClass(X,z);
% figure;
% plot(llh);