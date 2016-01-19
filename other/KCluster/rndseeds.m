function [S, idx] = rndseeds(X, k)
% Random pick k samples from X.
%   X: d x n data matrix
%   k: number of seeds
% Written by Michael Chen (sth4nth@gmail.com).
n = size(X,2);
idx = randsample(n,k);
S = X(:,idx);