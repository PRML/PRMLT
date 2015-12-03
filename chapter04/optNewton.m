function [w, llh, U] = optNewton(X, t, lambda, w)
% Newton-Raphson (second order) opitimzation method
% Written by Mo Chen (sth4nth@gmail.com).
[d,n] = size(X);
tol = 1e-4;
maxiter = 100;
llh = -inf(1,maxiter);

idx = (1:d)';
dg = sub2ind([d,d],idx,idx);
h = ones(1,n);
h(t==0) = -1;
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
    while true      % line search
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
