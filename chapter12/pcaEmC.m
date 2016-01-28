function [W, Z, err] = pcaEmC(X, p)
% Perform Constrained EM like algorithm for PCA.
%   X: d x n data matrix
%   p: dimension of target space
% Reference: 
%   A Constrained EM Algorithm for Principal Component Analysis by Jong-Hoon Ahn & Jong-Hoon Oh
% Written by Mo Chen (sth4nth@gmail.com).

[d,n] = size(X);
mu = mean(X,2);
X = bsxfun(@minus,X,mu);
W = rand(d,p); 

tol = 1e-6;
err = inf;
maxIter = 200;
for iter = 1:maxIter
    Z = tril(W'*W)\(W'*X);
    W = (X*Z')/triu(Z*Z');

    last = err;
    E = X-W*Z;
    err = E(:)'*E(:)/n;
    if abs(last-err)<err*tol; break; end;
end
fprintf('Converged in %d steps.\n',iter);