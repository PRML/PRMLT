function [model, llh] = rvmRegEbEm(X, t, alpha, beta)
% Relevance Vector Machine (ARD sparse prior) for regression
% training by empirical bayesian (type II ML) using standard EM update 
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 3
    alpha = 0.02;
    beta = 0.5;
end
% xbar = mean(X,2);
% tbar = mean(t,2);
% X = bsxfun(@minus,X,xbar);
% t = bsxfun(@minus,t,tbar);
n = size(X,2);
X = [X;ones(1,n)];
d = size(X,1);

% XX = X*X';
% Xt = X*t';

alpha = alpha*ones(d,1);
tol = 1e-4;
maxiter = 500;
llh = -inf(1,maxiter+1);
infinity = 1e8;
index = 1:d;
for iter = 2 : maxiter
    nz = alpha < infinity;   % nonzeros
    index = index(nz);
    alpha = alpha(nz);
    X = X(nz,:);
    
    S = inv(beta*(X*X')+diag(alpha));
    % E-step
    w = beta*S*X*t';   % E[w]     % 7.82
    w2 = m.^2+diag(S);       % E[w^2]
    e = sum((t-m'*X).^2);

%     logdetS = -2*sum(log(diag(V)));    
%     llh(iter) = 0.5*(sum(log(alpha))+n*log(beta)-beta*e-logdetS-dot(alpha,w2)-n*log(2*pi)); 
%     if abs(llh(iter)-llh(iter-1)) < tol*llh(iter-1); break; end

    % M-step
    alpha = 1./w2;    % 9.67
    beta = n/(e+sum(w2));    % 9.68 is wrong
end
llh = llh(2:iter);


model.index = index;
model.w0 = w0;
model.w = w;
model.alpha = alpha;
model.beta = beta;
