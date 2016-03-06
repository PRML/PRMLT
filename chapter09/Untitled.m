close all; clear;
d = 2;
k = 3;
n = 1000000;
[X,z,mu] = mixBernRnd(d,k,n);
model = nbBern(X,z);
mu
model.mu
% [label,model,llh] = mixBernEm(X,k);
% plot(llh);
% d = 2;
% n = 1000;
% mu = rand(d,1);
% X = bsxfun(@le,rand(d,n), mu);
% 
% m = mean(X,2)
% mu