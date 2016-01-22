
%% demo for sequential update Gaussian 
close all; clear;
d = 2;
n = 100;
X = randn(d,n);
x = randn(d,1);

mu = mean(X,2);
Xo = bsxfun(@minus,X,mu);
Sigma = Xo*Xo'/n;
p1 = logGauss(x,mu,Sigma);

gauss = Gaussian(X(:,3:end)).addSample(X(:,1)).addSample(X(:,2)).addSample(X(:,3)).delSample(X(:,3));
p2 = gauss.logPdf(x);
abs(p1-p2)
%%