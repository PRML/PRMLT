% demos for ch10
% chapter10/12: prediction functions for VB
%% Variational Bayesian for linear\RVM regression
clear; close all;

d = 100;
beta = 1e-1;
X = rand(1,d);
w = randn;
b = randn;
t = w'*X+b+beta*randn(1,d);
x = linspace(min(X),max(X),d);   % test data

[model,llh] = linRegVb(X,t);
% [model,llh] = rvmRegVb(X,t);
plot(llh);
[y, sigma] = linRegPred(model,x,t);
figure
plotCurveBar(x,y,sigma);
hold on;
plot(X,t,'o');
hold off
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

