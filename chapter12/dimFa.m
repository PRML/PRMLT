function model = dimFa(X, p)
% Perform EM algorithm for factor analysis model
%   X: d x n data matrix
%   p: dimension of target space
% Reference: Pattern Recognition and Machine Learning by Christopher M. Bishop 
% Written by Michael Chen (sth4nth@gmail.com).
[d,n] = size(X);
mu = mean(X,2);
X = bsxfun(@minus,X,mu);

tol = 1e-8;
converged = false;
llh = -inf;

% initialize parameters
W = rand(d,p); 
invpsi = 1./rand(d,1);

% precompute quantities
I = eye(p);
r = dot(X,X,2);

U = bsxfun(@times,W,sqrt(invpsi));
M = U'*U+I;                     % M = W'*inv(Psi)*W+I
R = chol(M);
invM = R\(R'\I);
WinvPsiX = bsxfun(@times,W,invpsi)'*X;       % WinvPsiX = W'*inv(Psi)*X
while ~converged
    % E step
    Ez = invM*WinvPsiX;
    Ezz = n*invM+Ez*Ez';
    % end
    
    R = chol(Ezz);  
    XEz = X*Ez';
    
    % M step
    W = (XEz/R)/R';
    invpsi = n./(r-dot(W,XEz,2));
    % end

    % compute quantities needed
    U = bsxfun(@times,W,sqrt(invpsi));
    M = U'*U+I;                     % M = W'*inv(Psi)*W+I
    R = chol(M);
    invM = R\(R'\I);
    WinvPsiX = bsxfun(@times,W,invpsi)'*X;       % WinvPsiX = W'*inv(Psi)*X
    % end
    
    % likelihood
    last = llh;
    logdetC = 2*sum(log(diag(R)))-sum(log(invpsi));              % log(det(C))
    Q = R'\WinvPsiX;
    trinvCS = (r'*invpsi-dot(Q(:),Q(:)))/n;  % trace(inv(C)*S)
    llh = -n*(d*log(2*pi)+logdetC+trinvCS)/2;
    % end
    converged = abs(llh-last) < tol*abs(llh);   % check likelihood for convergence
end
psi = 1./invpsi;

model.W = W;
model.mu = mu;
model.psi = psi;
model.llh = llh;
