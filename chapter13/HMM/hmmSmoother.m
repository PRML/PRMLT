function [gamma, alpha, beta, c] = hmmSmoother(x, model)
% HMM smoothing alogrithm (normalized forward-backward or normalized alpha-beta algorithm). This is a wrapper function which transform input and call underlying algorithm
% Unlike the method described in the book of PRML, the alpha and beta
% returned is the normalized.
% Computing unnormalized version alpha and beta is numerical unstable, which grows exponential fast to infinity.
% Input:
%   x: 1 x n integer vector which is the sequence of observations
%   model:  model structure
% Output:
%   gamma: k x n matrix of posterior gamma(t)=p(z_t,x_{1:T})
%   alpha: k x n matrix of posterior alpha(t)=p(z_t|x_{1:T})
%   beta: k x n matrix of posterior beta(t)=gamma(t)/alpha(t)
%   c: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
A = model.A;
E = model.E;
s = model.s;

n = size(x,2);
d = max(x);
X = sparse(x,1:n,1,d,n);
M = E*X;
[gamma, alpha, beta, c] = hmmSmoother_(M, A, s);
% [gamma,c] = hmmRecSmoother_(M, A, s);