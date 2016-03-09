function y = logGauss(X, mu, sigma)
% Compute log pdf of a Gaussian distribution.
% Input:
%   X: d x n data matrix
%   mu: d x 1 mean vector of Gaussian
%   sigma: d x d covariance matrix of Gaussian
% Output:
%   y: 1 x n probability density in logrithm scale y=log p(x)
% Written by Mo Chen (sth4nth@gmail.com).
[d,k] = size(mu);
if all(size(sigma)==d) && k==1   % one mu and one dxd sigma
    X = bsxfun(@minus,X,mu);
    [R,p]= chol(sigma);
    if p ~= 0
        error('ERROR: sigma is not PD.');
    end
    Q = R'\X;
    q = dot(Q,Q,1);  % quadratic term (M distance)
    c = d*log(2*pi)+2*sum(log(diag(R)));   % normalization constant
    y = -0.5*(c+q);
elseif size(sigma,1)==1 && size(sigma,2)==size(mu,2) % k mu and (k or one) scalar sigma
    X2 = repmat(dot(X,X,1)',1,k);
    D = bsxfun(@plus,X2-2*X'*mu,dot(mu,mu,1));
    q = bsxfun(@times,D,1./sigma);  % M distance
    c = d*(log(2*pi)+2*log(sigma));          % normalization constant
    y = -0.5*bsxfun(@plus,q,c);
end