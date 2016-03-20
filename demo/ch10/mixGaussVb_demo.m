
%% Variational Bayesian for Gaussian Mixture Model
close all; clear;
d = 2;
k = 3;
n = 2000;
[X,z] = mixGaussRnd(d,k,n);
plotClass(X,z);
m = floor(n/2);
X1 = X(:,1:m);
X2 = X(:,(m+1):end);
% VB fitting
[y1, model, L] = mixGaussVb(X1,10);
figure;
plotClass(X1,y1);
figure;
plot(L)
% Predict testing data
[y2, R] = mixGaussVbPred(model,X2);
figure;
plotClass(X2,y2);

