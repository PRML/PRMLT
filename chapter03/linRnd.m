function [X, t] = linRnd(d, n)
% Generate data from a linear model p(t|w,x)=G(w'x+w0,sigma), sigma=sqrt(1/beta) 
% where w and w0 are generated from Gauss(0,1),
%       beta is generated from Gamma(1,1),
%       X is generated form [0,1]
%   d: dimension of data
%   n: number of data
beta = gamrnd(10,10);   % need statistcs toolbox
X = rand(d,n);
w = randn(d,1);
w0 = randn(1,1);
err = randn(1,n)/sqrt(beta);
t = w'*X+w0+err;