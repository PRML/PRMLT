function [z, llh] = mrfMaxSum(W)
% Max-sum loopy belief propagation on Markov random field with discrete rvs
% Input:
%   W: n x n sparse weight matrix of a graph
% Output:
%   z: 1 x n vector of predicted label
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).

