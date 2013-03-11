function [V, A] = dimPcaLs(X, p)
% Perform Least Square iteration algorithm for PCA (by Sam Roweis).
%   X: d x n data matrix
%   p: dimension of target space
% Reference: 
%   Pattern Recognition and Machine Learning by Christopher M. Bishop 
%   EM algorithms for PCA and SPCA by Sam Roweis 
% Written by Michael Chen (sth4nth@gmail.com).
[d,n] = size(X);
X = bsxfun(@minus,X,mean(X,2));
W = rand(d,p); 

tol = 1e-8;
error = inf;
last = inf;
t = 0;
while ~(abs(last-error)<error*tol)
    t = t+1;
    Z = (W'*W)\(W'*X);
    W = (X*Z')/(Z*Z');

    last = error;
    E = X-W*Z;
    error = E(:)'*E(:)/n;
end
fprintf('Converged in %d steps.\n',t);
W = normalize(orth(W));
% [W,R] = qr(W,0); % qr() orthnormalize W which is faster than orth().
Z = W'*X;
Z = bsxfun(@minus,Z,mean(Z,2));  % for numerical purpose, not really necessary
[V,A] = eig(Z*Z');
[A,idx] = sort(diag(A),'descend');
V = V(:,idx);
V = W*V;
