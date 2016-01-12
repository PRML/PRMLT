function [model, llh] = rvmRegEm(X, t, alpha, beta)
% Relevance Vector Machine (ARD sparse prior) for regression
% training by empirical bayesian (type II ML) using standard EM update 
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 3
    alpha = 0.02;
    beta = 0.5;
end
xbar = mean(X,2);
tbar = mean(t,2);

X = bsxfun(@minus,X,xbar);
t = bsxfun(@minus,t,tbar);


alpha = alpha*ones(d,1);
tol = 1e-8;
maxiter = 500;
llh = -inf(1,maxiter+1);
index = 1:d;
for iter = 2 : maxiter
    nz = 1./alpha > tol ;   % nonzeros
    index = index(nz);
    alpha = alpha(nz);
    X = X(nz,:);
    
    A = beta*(X*X')+diag(alpha);
    % E-step
    m = beta*(A\(X*t'));   % E[m]     % 7.82
    m2 = m.^2;       % E[m^2]
    e2 = sum((t-m'*X).^2);

    logdetS = -2*sum(log(diag(V)));    
    llh(iter) = 0.5*(sum(log(alpha))+n*log(beta)-beta*e2-logdetS-dot(alpha,m2)-n*log(2*pi)); 
    if abs(llh(iter)-llh(iter-1)) < tol*llh(iter-1); break; end

    % M-step
    S = inv(A);
    alpha = 1./(m2+diag(S));    % 9.67
    
    trXSX = trace(X'*S*X);
    beta = n/(e2+trXSX);    % 9.68 is wrong
end
llh = llh(2:iter);


model.index = index;
model.w0 = w0;
model.m = m;
model.alpha = alpha;
model.beta = beta;
