function [model, llh] = regressRvmEbFpSvd(X, t, alpha, beta)
% Relevance Vector Machine regression training by empirical bayesian (ARD)
% using fix point update (Mackay update) with SVD
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

[U,S] = svd(X,'econ'); % X=U*S*V'
s = diag(S).^2;
UXt = U'*(X*t');

maxiter = 100;
llh = -inf(1,maxiter+1);
tol = 1e-2;
for iter = 2 : maxiter
    h = s+alpha/beta;
    m = U*(UXt./h);
    m2 = m.^2;
    err = sum((t-m'*X).^2);

    logdetS = sum(log(beta*h));
    llh(iter) = 0.5*(sum(log(alpha))+n*log(beta)-beta*err-logdetS-dot(alpha,m2)-n*log(2*pi)); 
    if abs(llh(iter)-llh(iter-1)) < tol*llh(iter-1); break; end

    V = bsxfun(@times,U,1./sqrt(h));
    dgS = dot(V,V,2);  
    gamma = 1-alpha.*dgS;   % 7.89
    alpha = gamma./m2;           % 7.87
    beta = (n-sum(gamma))/err;    % 7.88
end
llh = llh(2:iter);
m0 = tbar-dot(m,xbar);
model.b = m0;
model.w = m;
model.alpha = alpha;
model.beta = beta;
