function [y, sigma, p] = knRegPred(model, Xt, t)
% Prediction for Gaussian Process (kernel) regression model
%   model: trained model structure
%   Xt: d x n testing data
%   t (optional): 1 x n testing response
% Written by Mo Chen (sth4nth@gmail.com).
kn = model.kn;
a = model.a;
X = model.X;
tbar = model.tbar;
Kt = knCenter(kn,X,X,Xt);
y = a'*Kt+tbar;
%% probability prediction 
if nargout > 1
    alpha = model.alpha;
    beta = model.beta;
    U = model.U;
    XU = U'\Kt;
    sigma = sqrt(1/beta+(knCenter(kn,X,Xt)-dot(XU,XU,1))/alpha); 
end

if nargin == 3 && nargout == 3
    p = exp(logGauss(t,y,sigma));
end