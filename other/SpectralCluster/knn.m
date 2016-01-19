function [D, N] = knn(X, Y, k)
% Find k nearest neighbors in Y of each sample in X.
% Written by Michael Chen (sth4nth@gmail.com).
D = sqdistance(Y, X);
[D, N] = sort(D);
N = N(2:(1+k),:);
D = D(2:(1+k),:);
