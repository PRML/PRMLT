function [model, energy] = dimPcaVb(X, q, prior)
% Perform EM algorithm to maiximize likelihood of probabilistic PCA model.
%   X: m x n data matrix
%   q: dimension of target space
% Reference: 
%   Pattern Recognition and Machine Learning by Christopher M. Bishop 
%   Probabilistic Principal Component Analysis by Michael E. Tipping & Christopher M. Bishop
% Written by Michael Chen (sth4nth@gmail.com).

[m,n] = size(X);
if nargin < 3
    a0 = 1e-4;
    b0 = 1e-4;
    c0 = 1e-4;
    d0 = 1e-4;
else
    a0 = prior.a;
    b0 = prior.b;
    c0 = prior.c;
    d0 = prior.d;
end

if nargin < 2
    q = m-1;
end
tol = 1e-6;
maxIter = 500;
energy = -inf(1,maxIter);
% init parameters

a = a0+m/2;
c = c0+m*n/2;
Ealpha = 1e-4;
Ebeta = 1e-4;
EW = rand(q,m); 
EWo = bsxfun(@minus,EW,mean(EW,2));
EWW = EWo*EWo'/m+EW*EW';
I = eye(q);

mu = mean(X,2);
Xo = bsxfun(@minus, X, mu);
s = dot(Xo(:),Xo(:));
for iter = 2:maxIter  
%     q(z)
    CZ = inv(I+Ebeta*EWW);
    EZ = Ebeta*CZ*EW*Xo;
    EZZ = n*CZ+EZ*EZ';

%     q(w)
    A = diag(Ealpha);
    CW = inv(A+Ebeta*EZZ);
    EW = Ebeta*CW*EZ*Xo';
    EWW = m*CW+EW*EW';
    
%     q(alpha)
    b = b0+diag(EWW)/2;
    Ealpha = a./b;
    
%     q(beta)
    WZ = EW'*EZ;
    d = d0+(s-2*dot(Xo(:),WZ(:))+dot(EWW(:),EZZ(:)))/2;
    Ebeta = c/d;
    
%     q(mu)
%     Emu = Ebeta/(lambda+n*Ebeta)*sum(X-WZ,2);

%     lower bound
    KLalpha = -sum(a*log(b));
    KLbeta = -c*log(d);
    KLW = 0.5*m*log(det(CW));
    KLZ = 0.5*n*log(det(CZ));
    
    energy(iter) = KLalpha+KLbeta+KLW+KLZ;
    if energy(iter)-energy(iter-1) < tol*abs(energy(iter-1)); break; end   % check likelihood for convergence
end
energy = energy(2:iter);
% W = normalize(orth(W));
% % [W,R] = qr(W,0); % qr() orthnormalize W which is faster than orth().
% Z = W'*X;
% Z = bsxfun(@minus,Z,mean(Z,2));  % for numerical purpose, not really necessary
% [V,A] = eig(Z*Z');
% [A,idx] = sort(diag(A),'descend');
% V = V(:,idx);
% V = W*V;

% model.V = V;
model.a = a;
model.b = b;
model.c = c;
model.d = d;
model.mu = mu;


