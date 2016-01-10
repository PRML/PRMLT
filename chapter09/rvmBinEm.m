function [model, llh] = rvmBinEm(X, t, alpha)
% Relevance Vector Machine (ARD sparse prior) for binary classification
% training by empirical bayesian (type II ML) using standard EM update 
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 3
    alpha = 1;
end
n = size(X,2);
X = [X;ones(1,n)];
d = size(X,1);
alpha = alpha*ones(d,1);
weight = zeros(d,1);

tol = 1e-4;
maxiter = 100;
llh = -inf(1,maxiter);
infinity = 1e+10;
for iter = 2:maxiter
    used = alpha < infinity;
    a = alpha(used);
    w = weight(used); 
    [w,energy,U] = optLogitNewton(X(used,:),t,a,w);  
    w2 = w.^2;
    llh(iter) = energy(end)+0.5*(sum(log(a))-2*sum(log(diag(U)))-dot(a,w2)-n*log(2*pi)); % 7.114
    if abs(llh(iter)-llh(iter-1)) < tol*llh(iter-1); break; end
    V = inv(U);
    dgS = dot(V,V,2);
    alpha(used) = 1./(w2+dgS);    % 9.67
    weight(used) = w;
end
llh = llh(2:iter);

model.used = used;
model.w = w;                   % nonzero elements of weight
model.a = a;                   % nonzero elements of alpha
model.weght = weight;
model.alpha = alpha;
