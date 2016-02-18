% TODO:
%   1) prediction functions for vb reg and mix
%   2) modify mixGaussMix to compute bound inside each factor

% %% regression
% clear; close all;
% 
% d = 100;
% beta = 1e-1;
% X = rand(1,d);
% w = randn;
% b = randn;
% t = w'*X+b+beta*randn(1,d);
% x = linspace(min(X)-1,max(X)+1,d);   % test data
% 
% [model,llh] = linRegVb(X,t);
% % [model,llh] = rvmRegVb(X,t);
% figure
% plot(llh);
% [y, sigma] = linPred(model,x);
% figure;
% hold on;
% plotBand(x,y,2*sigma);
% plot(X,t,'o');
% plot(x,y,'r-');
% hold off

%% Variational Bayesian for Gaussian Mixture Model
close all; clear;
d = 2;
k = 3;
n = 200;
[X,label] = mixGaussRnd(d,k,n);
plotClass(X,label);
[y, model, L, L2] = mixGaussVb(X,10);
figure;
plotClass(X,y);
figure;
plot(L)
L(end)
L2(end)
L(end)-L2(end)
