function V = pcaEmC(X, p)
% Perform Constrained EM like algorithm for PCA.
%   X: d x n data matrix
%   p: dimension of target space
% Reference: A Constrained EM Algorithm for Principal Component Analysis by Jong-Hoon Ahn & Jong-Hoon Oh
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
    Z = tril(W'*W)\(W'*X);
    W = (X*Z')/triu(Z*Z');

    last = error;
    E = X-W*Z;
    error = E(:)'*E(:)/n;
end
V = normalize(W);
fprintf('Converged in %d steps.\n',t);