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
d = size(X,1);
tol = 1e-4;
maxiter = 100;
llh = -inf(1,maxiter);
h = 2*t-1;
w = rand(d,1);
a = w'*X;
for iter = 2:maxiter
    y = sigmoid(a);                     % 4.87
    r = y.*(1-y);                       % 4.98
    Xw = bsxfun(@times, X, sqrt(r));
    H = Xw*Xw'+lambda*eye(d);           % 4.97
    g = X*(y-t)'+lambda*w;              % 4.96
    w = w-H\g; 
    a = w'*X;   
    llh(iter) = -sum(log1pexp(-h.*a))-0.5*lambda*dot(w,w); % 4.89
    if llh(iter)-llh(iter-1) < tol; break; end
end
llh = llh(2:iter);
model.w = w;
