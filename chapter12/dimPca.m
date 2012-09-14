function [V, A] = dimPca(X, p)
% Perform standard PCA (spectral method).
%   X: d x n data matrix
%   p: dimension of target space (p>=1) or ratio (0<p<1)
% Written by Michael Chen (sth4nth@gmail.com).

opts.disp = 0;
opts.issym = 1;
opts.isreal = 1;
opts.maxit = 500;

[d,n] = size(X);
if nargin == 1
    p = min(d,n);
end

X = bsxfun(@minus,X,mean(X,2));
if 0<p && p<1       % given ratio
    [V,A] = svd(X,'econ');
    A = diag(A).^2;
    
    S = cumsum(A);
    idc = (S/S(end))<=p;
    V = V(:,idc);
    A = A(idc);
elseif p >= min(d,n) % full pca
    [V,A] = svd(X,'econ');
    A = diag(A).^2;
elseif d <= n       % covariance based pca
    [V,A] = eigs(X*X',p,'la',opts); 
    A = diag(A);
elseif d > n                % inner product based pca
    [U,A] = eigs(X'*X,p,'la',opts); 
    A = diag(A);
    V = X*bsxfun(@times,U,1./sqrt(A)');
end

