function [model, llh] = mixLogit(X, t, k)
% mixture of logistic regression model
n = size(X,2);
X = [X; ones(1,n)];
d = size(X,1);
z = ceil(k*rand(1,n));
R = full(sparse(1:n,z,1,n,k,n)); %  n x k

W = zeros(d,k);
w0 = zeros(1,k);
tol = 1e-6;
maxiter = 100;
llh = -inf(1,maxiter);
converged = false;
iter = 1;

t = t(:);
h = ones(n,1);
h(t==0) = -1;
A = bsxfun(@plus,X'*W,w0);
while ~converged && iter < maxiter
    iter = iter+1;
    % maximization
    nk = sum(R,1);
    alpha = nk/n;
    Y = sigmoid(A);
    for j = 1:k
        W(:,j) = newtonStep(X, t, Y(:,j), W(:,j), R(:,j));
    end

    % expectation
    A = bsxfun(@plus,X'*W,w0);
    logRho = -log1pexp(-bsxfun(@times,A,h));
    logRho = bsxfun(@plus,logRho,log(alpha));
    T = logsumexp(logRho,2);
    llh(iter) = sum(T)/n; % loglikelihood
    logR = bsxfun(@minus,logRho,T);
    R = exp(logR);
    
    converged = (llh(iter)-llh(iter-1)) < tol*abs(llh(iter));
end
llh = llh(2:iter);
model.alpha = alpha; % mixing coefficient
model.W = W;  % logistic model coefficent


function w = newtonStep(X, t, y, w, r)
% lambda = 1e-6;
v = y.*(1-y).*r;
H = bsxfun(@times,X,v')*X';%+lambda*eye(size(X,1));
s = (y-t).*r;
g = X*s;
w = w-H\g;

