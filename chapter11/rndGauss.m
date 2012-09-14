function x = rndGauss(mu,Sigma,n)
% Sampling from a Gaussian distribution.
% Written by Michael Chen (sth4nth@gmail.com).
if nargin == 2
    n = 1;
end
[V,err] = chol(Sigma);
if err ~= 0
    error('ERROR: sigma must be a symmetric positive semi-definite matrix.');
end
x = V'*randn(size(V,1),n)+repmat(mu,1,n);