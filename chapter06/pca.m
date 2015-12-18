function model = pca( X, p )
% Principal component analysis
% Written by Mo Chen (sth4nth@gmail.com).

Xo = bsxfun(@minus,X,mean(X,2));
S = Xo*Xo'/size(X,2);                   % 12.3
[U,A] = eig(S);                         % 12.5
[A,idx] = sort(diag(A),'descend');      
U = U(:,idx(1:p));
A = A(1:p);

model.U = U;
model.A = A;