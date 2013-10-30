function [model, llh, U] = logitReg(X, t, lambda)
% logistic regression for binary classification (Bernoulli likelihood)
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 3
    lambda = 1e-6;
end
[d,n] = size(X);
X = [X; ones(1,n)];
tol = 1e-4;
maxiter = 100;
llh = -inf(1,maxiter);

d = d+1;
idx = (1:d)';
dg = sub2ind([d,d],idx,idx);
h = ones(1,n);
h(t==0) = -1;
w = zeros(d,1);
z = w'*X;
for iter = 2:maxiter
    y = sigmoid(z);
    Xw = bsxfun(@times, X, sqrt(y.*(1-y)));
    H = Xw*Xw';
    H(dg) = H(dg)+lambda;
    U = chol(H);
    g = X*(y-t)'+lambda.*w;
    p = -U\(U'\g);
    wo = w;
    while true
        w = wo+p;
        z = w'*X;   
        llh(iter) = -sum(log1pexp(-h.*z))-0.5*sum(lambda.*w.^2);
        progress = llh(iter)-llh(iter-1);
        if progress < 0
            p = p/2;
        else
           break;
        end
    end
    if progress < tol
        break
    end
end
llh = llh(2:iter);
model.w = w;