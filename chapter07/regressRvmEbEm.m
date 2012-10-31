function [model, llh] = regressRvmEbEm(X, t, alpha, beta)
% Relevance Vector Machine regression training by empirical bayesian (ARD)
% using standard EM update 
if nargin < 3
    alpha = 0.02;
    beta = 0.5;
end
[d,n] = size(X);
alpha = alpha*ones(d,1);

xbar = mean(X,2);
tbar = mean(t,2);

X = bsxfun(@minus,X,xbar);
t = bsxfun(@minus,t,tbar);

XX = X*X';
Xt = X*t';

tol = 1e-4;
maxiter = 100;
llh = -inf(1,maxiter+1);
dg = sub2ind([d,d],1:d,1:d)';

infinity = 1e+10;
for iter = 2 : maxiter
    used = alpha < infinity;
    d = sum(used);
    alphaUsed = alpha(used);
    S = beta*XX(used,used);
    S(dg) = S(dg)+alphaUsed;
    U = chol(S);   
    V = U\eye(d);    
    w = beta*(V*(V'*Xt(used)));               % 7.82    
    w2 = w.^2;
    err = sum((t-w'*X(used,:)).^2);
    
    logdetS = -2*sum(log(diag(V)));    
    llh(iter) = 0.5*(sum(log(alphaUsed))+n*log(beta)-beta*err-logdetS-dot(alphaUsed,w2)-n*log(2*pi)); 
    if abs(llh(iter)-llh(iter-1)) < tol; break; end

    dgS = dot(V,V,2);
    alpha = 1./(w2+dgS);    % 9.67

    gamma = 1-alphaUsed.*dgS;   % 7.89
    beta = n/(err+sum(gamma)/beta);    % 9.68
end
llh = llh(2:iter);

b = tbar-dot(w,xbar(used));

model.used = used;
model.b = b;
model.w = w;
model.alpha = alpha;
model.beta = beta;
