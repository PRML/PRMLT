function [model, llh] = logitBin(X, y, lambda)
% Logistic regression for binary classification optimized by Newton-Raphson method.
% Input:
%   X: d x n data matrix
%   y: 1 x n label (0/1)
%   lambda: regularization parameter
%   alpha: step size
% Output:
%   model: trained model structure
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 4
    alpha = 1e-1;
end
if nargin < 3
    lambda = 1e-4;
end
X = [X; ones(1,size(X,2))];
[d,n] = size(X);
tol = 1e-4;
epoch = 200;
llh = -inf(1,epoch);
w = rand(d,1);
for t = 2:epoch
    a = w'*X;
    llh(t) = (dot(a,y)-sum(log1pexp(a))-0.5*lambda*dot(w,w))/n; % 4.90
    if abs(llh(t)-llh(t-1)) < tol; break; end
    z = sigmoid(a);                     % 4.87
    g = X*(z-y)'+lambda*w;              % 4.96
    r = z.*(1-z);                       % 4.98
    Xw = bsxfun(@times, X, sqrt(r));
    H = Xw*Xw'+lambda*eye(d);           % 4.97
    w = w-alpha*(H\g);                  % 4.92
end
llh = llh(2:t);
model.w = w;
