function model = nbBern(X, t)
% Naive bayes classifier with indepenet Bernoulli.
% Input:
%   X: d x n data matrix
%   t: 1 x n label (1~k)
% Output:
%   model: trained model structure
% Written by Mo Chen (sth4nth@gmail.com).
k = max(t);
n = size(X,2);
E = sparse(t,1:n,1,k,n,n);
nk = full(sum(E,2));
w = nk/n;
mu = full(sparse(X)*E'*spdiags(1./nk,0,k,k));  

model.mu = mu;      % d x k means 
model.w = w;