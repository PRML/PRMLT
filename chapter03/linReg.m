function model = linReg(X, t, lambda)
% Fit linear regression model t=w'x+w0
% X: d x n data
% t: 1 x n response
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 3
    lambda = 0;
end
d = size(X,1);
xbar = mean(X,2);
tbar = mean(t,2);

X = bsxfun(@minus,X,xbar);
t = bsxfun(@minus,t,tbar);

S = X*X';
idx = (1:d)';
dg = sub2ind([d,d],idx,idx);
S(dg) = S(dg)+lambda;
% w = S\(X*t');
U = chol(S);
w = U\(U'\(X*t'));  % 3.15 & 3.28
w0 = tbar-dot(w,xbar);  % 3.19

model.w = w;
model.w0 = w0;
