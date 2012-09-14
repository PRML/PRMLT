function [model, llh] = regressEbFp(X, t, alpha, beta)
% Fit empirical Bayesian linear model with Mackay fixed point method
% X: d x n data
% t: 1 x n response
if nargin < 3
    alpha = 0.02;
    beta = 0.5;
end
[d,n] = size(X);

xbar = mean(X,2);
tbar = mean(t,2);

X = bsxfun(@minus,X,xbar);
t = bsxfun(@minus,t,tbar);

C = X*X';
Xt = X*t';
dg = sub2ind([d,d],1:d,1:d);
I = eye(d);
tol = 1e-4;
maxiter = 100;
llh = -inf(1,maxiter+1);
for iter = 2:maxiter
    A = beta*C;
    A(dg) = A(dg)+alpha;
    U = chol(A);
    V = U\I;

    w = beta*(V*(V'*Xt));
    w2 = dot(w,w);
    err = sum((t-w'*X).^2);
    
    logdetA = 2*sum(log(diag(U)));    
    llh(iter) = 0.5*(d*log(alpha)+n*log(beta)-alpha*w2-beta*err-logdetA-n*log(2*pi)); 
    if llh(iter)-llh(iter-1) < tol*abs(llh(iter-1)); break; end
    
    trS = dot(V(:),V(:));
    gamma = d-alpha*trS;
    alpha = gamma/w2;
    beta = (n-gamma)/err;
end
b = tbar-dot(w,xbar);

llh = llh(2:iter);
model.b = b;
model.w = w;
model.alpha = alpha;
model.beta = beta;
model.xbar = xbar;
model.V = V;