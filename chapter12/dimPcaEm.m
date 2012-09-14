function model = dimPcaEm(X, p)
% Perform EM algorithm to maiximize likelihood of probabilistic PCA model.
%   X: d x n data matrix
%   p: dimension of target space
% Reference: 
%   Pattern Recognition and Machine Learning by Christopher M. Bishop 
%   Probabilistic Principal Component Analysis by Michael E. Tipping & Christopher M. Bishop
% Written by Michael Chen (sth4nth@gmail.com).

[d,n] = size(X);
mu = mean(X,2);
X = bsxfun(@minus,X,mu);

tol = 1e-6;
converged = false;
llh = -inf;

% init parameters
W = rand(d,p); 
s = rand;

% precompute quantities
di = sub2ind([p,p],1:p,1:p);  % diagonal index
I = eye(p);
r = dot(X(:),X(:)); % total norm of X

% M = W'*W+s*I;
M = W'*W;
M(di) = M(di)+s;

R = chol(M);
invM = R\(R'\I);
WX = W'*X;
while ~converged
    % E step
    Ez = invM*(WX);
    Ezz = n*s*invM+Ez*Ez'; % n*s because we are dealing with all n E[zi*zi']
    
    % M step
    R = chol(Ezz);  
    W = ((X*Ez')/R)/R';
    WR = W*R';
    s = (r-2*dot(Ez(:),WX(:))+dot(WR(:),WR(:)))/(n*d);
    
%     M = W'*W+s*I;
    M = W'*W;
    M(di) = M(di)+s;

    R = chol(M);
    invM = R\(R'\I);
    WX = W'*X;
    
    % likelihood
    last = llh;
    logdetC = 2*sum(log(diag(R)))+(d-p)*log(s);
    U = R'\WX;
    trinvCS = (r-dot(U(:),U(:)))/(s*n);
    llh = -n*(d*log(2*pi)+logdetC+trinvCS)/2;

    converged = abs(llh-last) < tol*abs(llh);   % check likelihood for convergence
end
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
model.llh = llh;

