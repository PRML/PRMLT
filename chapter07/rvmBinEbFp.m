function [model, llh] = rvmBinEbFp(X, t, alpha)
% Relevance Vector Machine (ARD sparse prior) for binary classification
% training by empirical bayesian (type II ML) using fix point update (Mackay update)
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 3
    alpha = 1;
end
n = size(X,2);
X = [X;ones(1,n)];
d = size(X,1);
alpha = alpha*ones(d,1);
m = zeros(d,1);

tol = 1e-3;
maxiter = 100;
llh = -inf(1,maxiter);
index = 1:d;
for iter = 2:maxiter
    % remove zeros
    nz = 1./alpha > tol;    % nonzeros
    index = index(nz);
    alpha = alpha(nz);
    X = X(nz,:);
    m = m(nz); 
    
    [m,e,U] = logitNewton(X,t,alpha,m);  
    
    m2 = m.^2;
    llh(iter) = e(end)+0.5*(sum(log(alpha))-2*sum(log(diag(U)))-dot(alpha,m2)-n*log(2*pi)); % 7.114
    if abs(llh(iter)-llh(iter-1)) < tol*llh(iter-1); break; end
    V = inv(U);
    dgS = dot(V,V,2);
    alpha = (1-alpha.*dgS)./m2;       % 7.89 & 7.87
end
llh = llh(2:iter);

model.index = index;
model.w = m;                  
model.alpha = alpha;


function [w, llh, U] = logitNewton(X, t, lambda, w)
% logistic regression for binary classification (Bernoulli likelihood)
% Written by Mo Chen (sth4nth@gmail.com).
[d,n] = size(X);
tol = 1e-4;
maxiter = 100;
llh = -inf(1,maxiter);

idx = (1:d)';
dg = sub2ind([d,d],idx,idx);
h = ones(1,n);
h(t==0) = -1;
z = w'*X;
for iter = 2:maxiter
    y = sigmoid(z);
    Xw = bsxfun(@times, X, sqrt(y.*(1-y)));
    H = Xw*Xw';
    H(dg) = H(dg)+lambda;
    U = chol(H);
    g = X*(y-t)'+lambda.*w;
    p = -U\(U'\g);
    wo = w;
    while true      % line search
        w = wo+p;
        z = w'*X;   
        llh(iter) = -sum(log1pexp(-h.*z))-0.5*sum(lambda.*w.^2);
        progress = llh(iter)-llh(iter-1);
        if progress < 0
            p = p/2;
        else
           break;
        end
    end
    if progress < tol
        break
    end
end
llh = llh(2:iter);
