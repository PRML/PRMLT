function [model, llh] = ppcaEm(X, q)
% Perform EM algorithm to maiximize likelihood of probabilistic PCA model.
%   X: m x n data matrix
%   q: dimension of target space
% Reference: 
%   Pattern Recognition and Machine Learning by Christopher M. Bishop 
%   Probabilistic Principal Component Analysis by Michael E. Tipping & Christopher M. Bishop
% Written by Mo Chen (sth4nth@gmail.com).
[m,n] = size(X);
mu = mean(X,2);
X = bsxfun(@minus,X,mu);

tol = 1e-4;
maxiter = 500;
llh = -inf(1,maxiter);
idx = (1:q)';
dg = sub2ind([q,q],idx,idx);
I = eye(q);
r = dot(X(:),X(:)); % total norm of X

W = rand(m,q); 
s = rand;
for iter = 2:maxiter
    M = W'*W;
    M(dg) = M(dg)+s;
    U = chol(M);
    invM = U\(U'\I);
    WX = W'*X;
    
    % likelihood
    logdetC = 2*sum(log(diag(U)))+(m-q)*log(s);
    T = U'\WX;
    trInvCS = (r-dot(T(:),T(:)))/(s*n);
    llh(iter) = -n*(m*log(2*pi)+logdetC+trInvCS)/2;
    if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter-1)); break; end   % check likelihood for convergence
    
    % E step
    Ez = invM*(WX);
    Ezz = n*s*invM+Ez*Ez'; % n*s because we are dealing with all n E[zi*zi']
    
    % M step
    U = chol(Ezz);  
    W = ((X*Ez')/U)/U';
    WR = W*U';
    s = (r-2*dot(Ez(:),WX(:))+dot(WR(:),WR(:)))/(n*m);
end
llh = llh(2:iter);
% W = normalize(orth(W));
% % [W,U] = qr(W,0); % qr() orthnormalize W which is faster than orth().
% Z = W'*X;
% Z = bsxfun(@minus,Z,mean(Z,2));  % for numerical purpose, not really necessary
% [V,A] = eig(Z*Z');
% [A,idx] = sort(diag(A),'descend');
% V = V(:,idx);
% V = W*V;
% model.V = V;
model.W = W;
model.mu = mu;
model.sigma = s;