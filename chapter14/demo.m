% TODO: demo for mixLogitReg
%% Demo for mixture of linear regression
% close all; clear
% d = 1;
% k = 2;
% n = 500;
% [X,y] = mixLinRnd(d,k,n);
% plot(X,y,'.');
% [label,model,llh] = mixLinReg(X, y, k);
% plotClass([X;y],label);
% figure
% plot(llh);

%% Demo for adaboost
close all; clear
d = 2;
k = 2;
n = 500;
[X,t] = kmeansRnd(d,k,n);
plotClass(X,t);
t = t-1;

model = adaboost(X,t);
