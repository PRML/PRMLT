function [model, llh] = classLogitMul(X, t, lambda, method)
% logistic regression for multiclass problem (Multinomial likelihood)
if nargin < 4
    method = 1;
end

if nargin < 3
    lambda = 1e-4;
end

X = [X; ones(1,size(X,2))];

if method == 1
    [W, llh] = NewtonSolver(X, t, lambda);
else
    [W, llh] = blockNewtonSolver(X, t, lambda);
end
model.W = W;

function [W, llh] = NewtonSolver(X, t, lambda)

[d,n] = size(X);
k = max(t);

tol = 1e-4;
maxiter = 100;
llh = -inf(1,maxiter);
converged = false;
iter = 1;
dk = d*k;
dg = sub2ind([dk,dk],1:dk,1:dk);
T = sparse(1:n,t,1,n,k,n);
W = zeros(d,k);
HT = zeros(d,k,d,k);
while ~converged && iter < maxiter
    iter = iter+1;
    Z = X'*W;
    logY = bsxfun(@minus,Z,logsumexp(Z,2));
    llh(iter) = dot(T(:),logY(:))-0.5*lambda*dot(W(:),W(:));
    converged = abs(llh(iter)-llh(iter-1)) < tol;
    
    Y = exp(logY);
    for i = 1:k
        for j = 1:k
            r = Y(:,i).*((i==j)-Y(:,j));
            HT(:,i,:,j) = bsxfun(@times,X,r')*X';
        end
    end
    G = X*(Y-T)+lambda*W;
    H = reshape(HT,dk,dk);
    H(dg) = H(dg)+lambda;
    W(:) = W(:)-H\G(:);
end
llh = llh(2:iter);

function [W, llh] = blockNewtonSolver(X, t, lambda)

[d,n] = size(X);
k = max(t);

dg = sub2ind([d,d],1:d,1:d);
tol = 1e-4;
maxiter = 100;
llh = -inf(1,maxiter);
converged = false;
iter = 1;

T = sparse(1:n,t,1,n,k,n);
W = zeros(d,k);
Z = X'*W;
logY = bsxfun(@minus,Z,logsumexp(Z,2));
Y = exp(logY);
while ~converged && iter < maxiter
    iter = iter+1;
    for j = 1:k
        r = Y(:,j).*(1-Y(:,j));
        H = bsxfun(@times,X,r')*X';
        H(dg) = H(dg)+lambda;

        g = X*(Y(:,j)-T(:,j))+lambda*W(:,j);
        W(:,j) = W(:,j)-H\g;
        Z(:,j) = X'*W(:,j);
        logY = bsxfun(@minus,Z,logsumexp(Z,2));
        Y = exp(logY);
    end
    
    llh(iter) = dot(T(:),logY(:))-0.5*lambda*dot(W(:),W(:));
    converged = abs(llh(iter)-llh(iter-1)) < tol;
end
llh = llh(2:iter);
