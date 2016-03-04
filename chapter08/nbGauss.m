function model = nbGauss(X, t)
% Naive bayes classifier with indepenet Gauss
% Input:
%   X: d x n data matrix
%   t: 1 x n label (1~k)
% Output:
%   model: trained model structure
% Written by Mo Chen (sth4nth@gmail.com).
n = size(X,2);
k = max(t);
E = sparse(t,1:n,1,k,n,n);
nk = sum(E,2);
a = nk/n;
z = spdiags(1./nk,0,k,k);
mu = X*E'*z;  
mm = bsxfun(@times,X*E',1./nk');
sigma = sqdist(mu,X)*z;

model.mu = mu;
model.sigma = sigma;
model.a = a;