function [z, llh] = bpSumProd(x, h, E)
% Sum-product loopy belief propagation on Markov random field with discrete rvs
% Input:
%   x: 1 x n vector of n observations
%   h: 1 x d vector of factors
%   E: d x n edge matrix of a bipartite graph to represent the factor graph (d factors, n nodes)
% Output:
%   z: 1 x n vector of predicted label
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
