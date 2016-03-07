function y = logGauss(X, mu, Sigma)
% Compute log pdf of a Gaussian distribution.
% Input:
%   X: d x n data matrix
%   mu: d x 1 mean vector of Gaussian
%   Sigma: d x d covariance matrix of Gaussian
% Output:
%   y: 1 x n probability density in logrithm scale y=log p(x)
% Written by Mo Chen (sth4nth@gmail.com).
[d,k] = size(mu);
assert(all(size(Sigma)==d) && k==1)   % one mu and one dxd Sigma
X = bsxfun(@minus,X,mu);
[R,p]= chol(Sigma);
if p ~= 0
    error('ERROR: Sigma is not PD.');
end
Q = R'\X;
q = dot(Q,Q,1);  % quadratic term (M distance)
c = d*log(2*pi)+2*sum(log(diag(R)));   % normalization constant
y = -0.5*(c+q);

