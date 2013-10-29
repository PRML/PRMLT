function model = knReg(X, t, lambda, kn)
% Gaussian process for regression
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 4
    kn = @knGauss;
end
K = knCenterize(kn,X);
tbar = mean(t);
U = chol(K+lambda*eye(size(X,2)));
a = U\(U'\(t(:)-tbar));  

model.kn = kn;
model.a = a;
model.X = X;
model.tbar = tbar;