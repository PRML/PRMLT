function [W, mu, psi, llh] = fa(X, m)
% Perform EM algorithm for factor analysis model
%   X: d x n data matrix
%   m: dimension of target space
% Reference: Pattern Recognition and Machine Learning by Christopher M. Bishop 
% Written by Mo Chen (sth4nth@gmail.com).
[d,n] = size(X);
mu = mean(X,2);
X = bsxfun(@minus,X,mu);

tol = 1e-4;
maxiter = 500;
llh = -inf(1,maxiter);

I = eye(m);
r = dot(X,X,2);

W = randn(d,m); 
lambda = 1./rand(d,1);
for iter = 2:maxiter
    % compute quantities needed
    T = bsxfun(@times,W,sqrt(lambda));
    M = T'*T+I;                     % M = W'*inv(Psi)*W+I
    U = chol(M);
    G = U\(U'\I);                                   % 12.68
    WInvPsiX = bsxfun(@times,W,lambda)'*X;       % WInvPsiX = W'*inv(Psi)*X
    
    % likelihood
    logdetC = 2*sum(log(diag(U)))-sum(log(lambda));              % log(det(C))
    Q = U'\WInvPsiX;
    trInvCS = (r'*lambda-dot(Q(:),Q(:)))/n;  % trace(inv(C)*S)
    llh(iter) = -n*(d*log(2*pi)+logdetC+trInvCS)/2;
    if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter-1)); break; end   % check likelihood for convergence
    
    % E step
    Ez = G*WInvPsiX;                                         % 12.66
    Ezz = n*G+Ez*Ez';                                        % 12.67
    
    % M step    
    U = chol(Ezz);  
    XEz = X*Ez';
    W = (XEz/U)/U';                                         % 12.69
    lambda = n./(r-dot(W,XEz,2));                           % 12.70
end
llh = llh(2:iter);
psi = 1./lambda;