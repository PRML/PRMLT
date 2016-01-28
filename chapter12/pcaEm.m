function [W, Z, mu, err] = pcaEm(X, m)
% Perform EM-like algorithm for PCA (by Sam Roweis).
%   X: d x n data matrix
%   m: dimension of target space
% Reference: 
%   Pattern Recognition and Machine Learning by Christopher M. Bishop 
%   EM algorithms for PCA and SPCA by Sam Roweis 
% Written by Mo Chen (sth4nth@gmail.com).
[d,n] = size(X);
mu = mean(X,2);
X = bsxfun(@minus,X,mu);
W = rand(d,m); 

tol = 1e-6;
err = inf;
maxIter = 200;
for iter = 1:maxIter
    Z = (W'*W)\(W'*X);             % 12.58
    W = (X*Z')/(Z*Z');              % 12.59

    last = err;
    E = X-W*Z;
    err = E(:)'*E(:)/n;
    if abs(last-err)<err*tol; break; end;
end
fprintf('Converged in %d steps.\n',iter);

