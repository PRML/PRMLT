function [X, z, center] = kmeansRnd(d, k, n)
% Sampling from a Gaussian mixture distribution with common variances (kmeans model).
% Written by Michael Chen (sth4nth@gmail.com).
a = 1;
b = 6*nthroot(k,d);

X = randn(d,n);
w = dirichletRnd(ones(k,a));
z = discreteRnd(w,n);
E = full(sparse(z,1:n,1,k,n,n));
center = rand(d,k)*b;
X = X+center*E;