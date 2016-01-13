close all; clear
%% Demo for mixture of linear regression
d = 1;
k = 2;
n = 500;
[X,y] = mixLinRnd(d,k,n);
plot(X,y,'.');
[label,model,llh] = mixLinReg(X, y, k);
plotClass([X;y],label);
figure
plot(llh);

%%
% [X, y] = rndKmeans(2,3,1000);
% [label,L] = mixGaussVb(X, 10);
% plot(L);