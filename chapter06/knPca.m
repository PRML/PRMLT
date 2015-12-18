function [ R, Z, err, model] = knPca(X, d, kn)
% Kernel PCA
%   X: dxn data matrix 
%   d: target dimension
%   kn: kernel function
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 3
    kn = @knGauss;
end

K = kn(X,X);
K = knCenter(K);
[V,A] = eig(K);
[A,idx] = sort(diag(A),'descend');
V = V(:,idx(1:d))';
A = A(1:d);
R = bsxfun(@times,V,1./sqrt(A));
if nargout > 1    
    Z = bsxfun(@times,V,sqrt(A));
end
if nargout > 2
    err = diag(K)'-sum(Z.^2,1);
end