function [X, model] = ppcaRnd(m, d, n)
% Generate data from probabilistic PCA model
beta = randg;
Z = randn(m,n);
W = randn(d,m); 
mu = randn(d,1);
X = bsxfun(@times,W*Z,mu)+randn(d,n)/sqrt(beta);

model.W = W;
model.mu = mu;
model.beta = beta;