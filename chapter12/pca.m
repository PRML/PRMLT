function model = pca(X, p)
% Principal component analysis
%   X: dxn data matrix 
%   p: target dimension
% Written by Mo Chen (sth4nth@gmail.com).
xbar = mean(X,2);
Xo = bsxfun(@minus,X,xbar);
S = Xo*Xo'/size(X,2);                   % 12.3
[U,L] = eig(S);                         % 12.5
[L,idx] = sort(diag(L),'descend');      
U = U(:,idx(1:p));
L = L(1:p);

model.xbar = xbar;
model.U = U;
model.L = L;
