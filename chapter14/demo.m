%% Demo for mixture of linear regression
% close all; clear
% d = 1;
% k = 2;
% n = 500;
% [X,y] = mixLinRnd(d,k,n);
% plot(X,y,'.');
% [model,llh] = mixLinReg(X, y, k);
y_ = mixLinReg(X,y,k);
% plotClass([X;y],label);
% figure
% plot(llh);
%% Demo for mixture of logistic regression
d = 2;
k = 2;
n = 500;
[X,t] = kmeansRnd(d,k,n);

model = mixnLogitBin(X,t-1);
y = adaboostBinPred(model,X);
plotClass(X,y+1)
%% Demo for adaboost
d = 2;
k = 2;
n = 500;
[X,t] = kmeansRnd(d,k,n);
model = adaboostBin(X,t-1);
y = adaboostBinPred(model,X);
plotClass(X,y+1)

