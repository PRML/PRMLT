function [y, sigma, p] = knInfer(x, model)
% inference for kernel model
kn = model.kn;
a = model.a;
X = model.X;
tbar = model.tbar;
y = a'*knCenterize(kn,X,x)+tbar;

% sigma = sqrt(1/beta+dot(X,X,1));   % 3.59