function [model, llh] = pcaEm(X, q)
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

tol = 1e-6;
maxiter = 500;
llh = -inf(1,maxiter);

dg = sub2ind([q,q],1:q,1:q);  % diagonal index
I = eye(q);
r = dot(X(:),X(:)); % total norm of X

W = rand(m,q); 
s = rand;
for iter = 2:maxiter
    M = W'*W;
    M(dg) = M(dg)+s;
    R = chol(M);
    invM = R\(R'\I);
    WX = W'*X;
    
    % likelihood
    logdetC = 2*sum(log(diag(R)))+(m-q)*log(s);
    U = R'\WX;
    trInvCS = (r-dot(U(:),U(:)))/(s*n);
    llh(iter) = -n*(m*log(2*pi)+logdetC+trInvCS)/2;
    if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter-1)); break; end   % check likelihood for convergence
    
    % E step
    Ez = invM*(WX);
    Ezz = n*s*invM+Ez*Ez'; % n*s because we are dealing with all n E[zi*zi']
    
    % M step
    R = chol(Ezz);  
    W = ((X*Ez')/R)/R';
    WR = W*R';
    s = (r-2*dot(Ez(:),WX(:))+dot(WR(:),WR(:)))/(n*m);
end
llh = llh(2:iter);
% W = normalize(orth(W));
% % [W,R] = qr(W,0); % qr() orthnormalize W which is faster than orth().
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

