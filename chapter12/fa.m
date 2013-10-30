function [model, llh] = fa(X, q)
% Perform EM algorithm for factor analysis model
%   X: m x n data matrix
%   q: dimension of target space
% Reference: Pattern Recognition and Machine Learning by Christopher M. Bishop 
% Written by Mo Chen (sth4nth@gmail.com).
[m,n] = size(X);
mu = mean(X,2);
X = bsxfun(@minus,X,mu);

tol = 1e-4;
maxiter = 500;
llh = -inf(1,maxiter);

I = eye(q);
r = dot(X,X,2);

W = rand(m,q); 
invpsi = 1./rand(m,1);
for iter = 2:maxiter
    % compute quantities needed
    U = bsxfun(@times,W,sqrt(invpsi));
    M = U'*U+I;                     % M = W'*inv(Psi)*W+I
    R = chol(M);
    G = R\(R'\I);
    WinvPsiX = bsxfun(@times,W,invpsi)'*X;       % WinvPsiX = W'*inv(Psi)*X
    
    % likelihood
    logdetC = 2*sum(log(diag(R)))-sum(log(invpsi));              % log(det(C))
    Q = R'\WinvPsiX;
    trinvCS = (r'*invpsi-dot(Q(:),Q(:)))/n;  % trace(inv(C)*S)
    llh(iter) = -n*(m*log(2*pi)+logdetC+trinvCS)/2;
    if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter-1)); break; end   % check likelihood for convergence
    
    % E step
    Ez = G*WinvPsiX;
    Ezz = n*G+Ez*Ez';
    
    % M step    
    R = chol(Ezz);  
    XEz = X*Ez';
    W = (XEz/R)/R';
    invpsi = n./(r-dot(W,XEz,2));
end
llh = llh(2:iter);

model.W = W;
model.mu = mu;
model.psi = 1./invpsi;