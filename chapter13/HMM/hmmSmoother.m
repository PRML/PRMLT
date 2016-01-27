function [gamma, alpha, beta, c] = hmmSmoother(x, model)
% HMM smoothing alogrithm (normalized forward-backward or normalized alpha-beta algorithm)
A = model.A;
E = model.E;
s = model.s;

n = size(x,2);
d = max(x);
X = sparse(x,1:n,1,d,n);
M = E*X;
[gamma, alpha, beta, c] = hmmSmoother_(M, A, s);
