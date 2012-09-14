function model = regress(X, t, lambda)
% Fit linear regression model t=w'x+b
% X: d x n data
% t: 1 x n response
if nargin < 3
    lambda = 0;
end
d = size(X,1);
xbar = mean(X,2);
tbar = mean(t,2);

X = bsxfun(@minus,X,xbar);
t = bsxfun(@minus,t,tbar);

S = X*X';
dg = sub2ind([d,d],1:d,1:d);
S(dg) = S(dg)+lambda;
% w = S\(X*t');
R = chol(S);
w = R\(R'\(X*t'));  % 3.15 & 3.28
b = tbar-dot(w,xbar);  % 3.19

model.w = w;
model.b = b;
