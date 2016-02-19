function [model, llh] = logitBin(X, t, lambda)
% Logistic regression for binary classification optimized by Newton-Raphson method.
% Input:
%   X: d x n data matrix
%   t: 1 x n label (0/1)
%   lambda: regularization parameter
% Output:
%   model: trained model structure
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 3
    lambda = 1e-2;
end
X = [X; ones(1,size(X,2))];
[d,n] = size(X);
tol = 1e-4;
maxiter = 100;
llh = -inf(1,maxiter);
idx = (1:d)';
dg = sub2ind([d,d],idx,idx);
h = ones(1,n);
h(t==0) = -1;
w = zeros(d,1);
a = w'*X;
for iter = 2:maxiter
    y = sigmoid(a);                     % 4.87
    r = y.*(1-y);                       % 4.98
    Xw = bsxfun(@times, X, sqrt(r));
    H = Xw*Xw';                         % 4.97
    H(dg) = H(dg)+lambda;
    U = chol(H);
    g = X*(y-t)'+lambda.*w;             % 4.96
    p = -U\(U'\g);
    wo = w;                             % 4.92
    w = wo+p;   
    a = w'*X;   
    llh(iter) = -sum(log1pexp(-h.*a))-0.5*sum(lambda.*w.^2);  % 4.89
    incr = llh(iter)-llh(iter-1);
%     while incr < 0      % line search
%         p = p/2;
%         w = wo+p;
%         a = w'*X;   
%         llh(iter) = -sum(log1pexp(-h.*a))-0.5*sum(lambda.*w.^2);
%         incr = llh(iter)-llh(iter-1);
%     end
    if incr < tol; break; end
end
llh = llh(2:iter);
model.w = w;
