function [U, L, mu, err] = pca(X, m)
% Principal component analysis
% Input:
%   X: d x n data matrix 
%   m: target dimension
% Output:
%   U: d x m Projection matrix
%   L: m x 1 Eigen values
%   mu: d x 1 mean
%   err: recontruction error
% Written by Mo Chen (sth4nth@gmail.com).
mu = mean(X,2);
Xo = bsxfun(@minus,X,mu);
S = Xo*Xo'/size(X,2);                   % 12.3
[U,L] = eig(S);                         % 12.5
[L,idx] = sort(diag(L),'descend');      
err = sum(L)-sum(L(1:m));
U = U(:,idx(1:m));
L = L(1:m);

