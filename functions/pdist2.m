function D = pdist2(X1, X2)
% Pairwise square Euclidean distance between two sample sets
%   X1, X2: dxn1 dxn2 sample matrices
% Written by Mo Chen (sth4nth@gmail.com).
D = bsxfun(@plus,dot(X2,X2,1),dot(X1,X1,1)')-2*(X1'*X2);

