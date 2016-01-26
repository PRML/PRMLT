function K = knGauss(X, Y, s)
% Gaussian (RBF) kernel
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 3
    s = 1;
end

if nargin < 2 || isempty(Y)  
    K = ones(1,size(X,2));            % norm in kernel space
else
    D = bsxfun(@plus,dot(X,X,1)',dot(Y,Y,1))-2*(X'*Y);
    K = exp(D/(-2*s^2));
end

