function [model, llh] = classLogitBin(X, t, lambda)
% logistic regression for binary classification (Bernoulli likelihood)
if any(unique(t) ~= [0,1])
    error('t must be a 0/1 vector!');
end
if nargin < 3
    lambda = 1e-4;
end
[d,n] = size(X);
dg = sub2ind([d,d],1:d,1:d);
X = [X; ones(1,n)];
d = d+1;

tol = 1e-4;
maxiter = 100;
llh = -inf(1,maxiter);
converged = false;
iter = 1;


h = ones(1,n);
h(t==0) = -1;
w = zeros(d,1);
z = w'*X;
while ~converged && iter < maxiter
    iter = iter + 1;
    y = sigmoid(z);
    
    g = X*(y-t)'+lambda*w;
    r = y.*(1-y);
    R = spdiags(r(:),0,n,n);    
    H = X*R*X';
    H(dg) = H(dg)+lambda;
    
    p = -H\g;

    wo = w;
    w = wo+p;
    z = w'*X;   
    llh(iter) = -sum(log1pexp(-h.*z))-0.5*lambda*dot(w,w);
    converged = norm(p) < tol || abs(llh(iter)-llh(iter-1)) < tol;
    while ~converged && llh(iter) < llh(iter-1)
        p = 0.5*p;
        w = wo+p;
        z = w'*X;    
        llh(iter) = -sum(log1pexp(-h.*z))-0.5*lambda*dot(w,w);
        converged = norm(p) < tol || abs(llh(iter)-llh(iter-1)) < tol;
    end
end
llh = llh(2:iter);
model.w = w;
