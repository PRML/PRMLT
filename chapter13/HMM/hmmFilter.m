function [alpha, energy] = hmmFilter(x, model)
% HMM forward filtering algorithm. This is a wrapper function which transform input and call underlying algorithm
% Unlike the method described in the book of PRML, the alpha returned is the normalized version: alpha(t)=p(z_t|x_{1:t})
% Computing unnormalized version alpha(t)=p(z_t,x_{1:t}) is numerical unstable, which grows exponential fast to infinity.
% Input:
%   x: 1 x n integer vector which is the sequence of observations
%   model:  model structure
% Output:
%   alpha: k x n matrix of posterior alpha(t)=p(z_t|x_{1:t})
%   enery: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
A = model.A;
E = model.E;
s = model.s;

n = size(x,2);
d = max(x);
X = sparse(x,1:n,1,d,n);
M = E*X;
[alpha, energy] = hmmFilter_(M, A, s);