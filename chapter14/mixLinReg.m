function [model, label, llh] = mixLinReg(X, y, k, lambda)
% mixture of linear regression
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 4
    lambda = 1;
end
n = size(X,2);
X = [X;ones(1,n)]; % adding the bias term
d = size(X,1);
idx = (1:d)';
dg = sub2ind([d,d],idx,idx);
label = ceil(k*rand(1,n));  % random initialization
R = sparse(label,1:n,1,k,n,n);
tol = 1e-4;
maxiter = 200;
llh = -inf(1,maxiter);
lambda = lambda*ones(d,1);
W = zeros(d,k);
beta = 1;
for iter = 2 : maxiter
    % maximization
    nk = sum(R,2);
    alpha = nk/n;

    for j = 1:k
        Xw = bsxfun(@times,X,sqrt(R(j,:)));
        C = Xw*Xw';
        C(dg) = C(dg)+lambda;
        U = chol(C);
        W(:,j) = U\(U'\(X*(R(j,:).*y)'));  % 3.15 & 3.28
    end
    D = (bsxfun(@minus,W'*X,y)).^2;
    % expectation
    logRho = (-0.5)*beta*D;
    logRho = bsxfun(@plus,logRho,log(alpha));
    T = logsumexp(logRho,1);
    logR = bsxfun(@minus,logRho,T);
    R = exp(logR);
    llh(iter) = sum(T)/n;
    if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter)); break; end
end
[~,label] = max(R,[],1);
llh = llh(2:iter);

model.alpha = alpha; % mixing coefficient
model.beta = beta; % mixture component precision
model.W = W;  % linear model coefficent
