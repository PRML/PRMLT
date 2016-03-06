function [X, z, mu] = mixBernRnd(d, k, n)
% Generate samples from a Bernoulli mixture distribution.
% Input:
%   d: dimension of data
%   k: number of components
%   n: number of data
% Output:
%   X: d x n data matrix
%   z: 1 x n response variable
%   center: d x k centers of clusters
% Written by Mo Chen (sth4nth@gmail.com).
alpha = 1;

w = dirichletRnd(alpha,ones(1,k)/k);
z = discreteRnd(w,n);
mu = rand(1,k);

X = zeros(d,n);
for i = 1:k
    idx = z==i;
    X(:,idx) = rand(d,sum(idx)) < mu(k);
end
