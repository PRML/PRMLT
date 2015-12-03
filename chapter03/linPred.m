function [y, sigma, p] = linPred(model, X, t)
% Compute linear model reponse y = w'*X+w0 and likelihood
% model: trained model structure
% X: d x n testing data
% t (optional): 1 x n testing response
% Written by Mo Chen (sth4nth@gmail.com).
w = model.w;
w0 = model.w0;
y = w'*X+w0;
if nargin == 3
    beta = model.beta;
    if isfield(model,'V')   % V*V'=inv(S) 3.54
        U = model.V'*bsxfun(@minus,X,model.xbar);
        sigma = sqrt(1/beta+dot(U,U,1));   % 3.59
    else
        sigma = sqrt(1/beta);
    end
    p = exp(logGauss(t,y,sigma));
%     p = exp(-0.5*(((t-y)./sigma).^2+log(2*pi))-log(sigma));
end
