% demos for ch10
% chapter10/12: prediction functions for VB
%% regression
clear; close all;

d = 100;
beta = 1e-1;
X = rand(1,d);
w = randn;
b = randn;
t = w'*X+b+beta*randn(1,d);
x = linspace(min(X)-1,max(X)+1,d);   % test data

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
% close all; clear;
% d = 2;
% k = 3;
% n = 2000;
% [X,label] = mixGaussRnd(d,k,n);
% plotClass(X,label);
% [y, model, L] = mixGaussVb(X,10);
% figure;
% plotClass(X,y);
% figure;
% plot(L)


