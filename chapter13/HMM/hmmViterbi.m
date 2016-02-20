function z = hmmViterbi(x, model)
% Viterbi algorithm calculated in log scale to improve numerical stability.
% This is a wrapper function which transform input and call underlying algorithm
% Input:
%   x: 1 x n integer vector which is the sequence of observations
%   model:  model structure
% Output:
%   z: 1 x n latent state
% Written by Mo Chen (sth4nth@gmail.com).
A = model.A;
E = model.E;
s = model.s;

n = size(x,2);
d = max(x);
X = sparse(x,1:n,1,d,n);
M = E*X;
z = hmmViterbi_(M, A, s);
