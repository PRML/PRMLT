function [w, V, llh] = optLogitNewton(X, t, lambda, w)
% find the mode of logistic regression model using Newton method
% X: d x n data
% t: 1 x n 0/1 label
% A: d x d regularization penalty
% w: d x 1 initial value of w

[d,n] = size(X);

if nargin < 4
    w = zeros(d,1);
end
tol = 1e-4;
maxiter = 100;
llh = -inf(1,maxiter);
converged = false;
iter = 1;

h = ones(1,n);
h(t==0) = -1;

z = w'*X;
while ~converged && iter < maxiter
    iter = iter + 1;
    y = sigmoid(z);
    
    g = X*(y-t)'+lambda.*w;
    r = y.*(1-y);
    R = spdiags(r(:),0,n,n);    
    H = X*R*X'+diag(lambda);
    U = chol(H);
    p = -U\(U'\g);

    wo = w;
    w = wo+p;
    z = w'*X;   
    llh(iter) = -sum(log1pexp(-h.*z))-0.5*dot(lambda,w.^2);
    converged = norm(p) < tol || abs(llh(iter)-llh(iter-1)) < tol;
    while ~converged && llh(iter) < llh(iter-1)
        p = 0.5*p;
        w = wo+p;
        z = w'*X;    
        llh(iter) = -sum(log1pexp(-h.*z))-0.5*iter*dot(w,w);
        converged = norm(p) < tol || abs(llh(iter)-llh(iter-1)) < tol;
    end
end
llh = llh(iter);
V = U\eye(d);