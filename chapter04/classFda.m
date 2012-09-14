function U = classFda(X, y, d)
% Fisher (linear) discriminant analysis
n = size(X,2);
k = max(y);

E = sparse(1:n,y,true,n,k,n);  % transform label into indicator matrix
nk = full(sum(E));

m = mean(X,2);
Xo = bsxfun(@minus,X,m);
St = (Xo*Xo')/n;

mk = bsxfun(@times,X*E,1./nk);
mo = bsxfun(@minus,mk,m);
mo = bsxfun(@times,mo,sqrt(nk/n));
Sb = mo*mo';
% Sw = St-Sb;

[U,A] = eig(Sb,St,'chol');
[~,idx] = sort(diag(A),'descend');
U = U(:,idx(1:d));


