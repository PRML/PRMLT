function [z, llh] = bpMaxSum(x, h, E)
% Max-sum loopy belief propagation on a factor graph over discrete random variables
% Input:
%   x: 1 x n vector of n observations
%   h: 1 x d vector of factors
%   E: d x n edge matrix of a bipartite graph to represent the factor graph (d factors, n nodes)
% Output:
%   z: 1 x n vector of predicted label
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).

