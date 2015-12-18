function model = knPca(X, p, kn)
% Kernel PCA
%   X: dxn data matrix 
%   p: target dimension
%   kn: kernel function
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 3
    kn = @knGauss;
end
K = kn(X,X);
K = knCenter(K);
[V,L] = eig(K);
[L,idx] = sort(diag(L),'descend');
V = V(:,idx(1:p));
L = L(1:p)';
U = bsxfun(@times,V,1./sqrt(L));
if nargout > 1    
    Z = bsxfun(@times,V,sqrt(L));
end
model.V = V;
model.L = L;
model.X = X;