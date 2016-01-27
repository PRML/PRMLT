function [alpha, energy] = hmmFilter(x, model)
% HMM forward filtering algorithm
% Unlike the method described in the book of PRML, the alpha returned is the normalized version: alpha(t)=p(z_t|x_{1:t})
% The unnormalized version is numerical unstable. alpha(t)=p(z_t,x_{1:t}) grows exponential fast to infinity.
% Written by Mo Chen (sth4nth@gmail.com).
A = model.A;
E = model.E;
s = model.s;

n = size(x,2);
d = max(x);
X = sparse(x,1:n,1,d,n);
M = E*X;
[alpha, energy] = hmmFilter_(M, A, s);