function [y, sigma, p] = knRegPred(model, Xt, t)
% Prediction for Gaussian Process (kernel) regression model
% Written by Mo Chen (sth4nth@gmail.com).
kn = model.kn;
a = model.a;
X = model.X;
tbar = model.tbar;
y = a'*knCenterize(kn,X,Xt)+tbar;
if nargout > 1
    beta = model.beta;
    if isfield(model,'U')
        U = model.U;
        Xo = bsxfun(@minus,X,model.xbar);
        XU = U'\Xo;
        sigma = sqrt(1/beta+dot(XU,XU,1));
        
        sigma = sqrt(c-k'*C^-1*k);
    else
        sigma = sqrt(1/beta);   % 6.67
    end
    if nargin == 3 && nargout == 3
        p = exp(logGauss(t,y,sigma));
    end

end
