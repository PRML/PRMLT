function [argmax, prob] = hmmViterbi(x, model)
% Viterbi algorithm
A = model.A;
E = model.E;
s = model.s;

n = size(x,2);
d = max(x);
X = sparse(x,1:n,1,d,n);
M = E*X;
[argmax, prob] = hmmViterbi_(M, A, s);
