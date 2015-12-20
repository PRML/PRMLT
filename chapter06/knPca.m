function model = knPca(X, p, kn)
% Kernel PCA
%   X: dxn data matrix 
%   p: target dimension
%   kn: kernel function
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 3
    kn = @knGauss;
end
K = knCenter(kn,X);
[V,L] = eig(K);
[L,idx] = sort(diag(L),'descend');
V = V(:,idx(1:p));
L = L(1:p);

model.kn = kn;
model.V = V;
model.L = L;
model.X = X;