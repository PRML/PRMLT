function K = knGauss(X, Y, s)
% Gaussian (RBF) kernel
if nargin < 3
    s = 1;
end

D = bsxfun(@plus,dot(X,X,1)',dot(Y,Y,1))-2*(X'*Y);
K = exp(D/(-2*s^2));
