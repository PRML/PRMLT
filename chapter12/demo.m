% demo
m = 10;
n = 1000;
X = randn(m,n);
mu = mean(X,2);
Xo = bsxfun(@minus,X,mu);
[U,S,V] = svd(Xo,'econ');
r = rand(m,1).^8;

S = S.*diag(r);
Xo = U*S*V';
X = bsxfun(@plus,Xo,mu);

%%
[model,energy] = dimPcaVb(X);