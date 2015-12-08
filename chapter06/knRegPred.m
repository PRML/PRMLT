function [y, sigma, p] = knRegPred(model, x, t)
% Prediction for kernel regression model
% Written by Mo Chen (sth4nth@gmail.com).
kn = model.kn;
a = model.a;
X = model.X;
tbar = model.tbar;
y = a'*knCenterize(kn,X,x)+tbar;
if nargin == 3
    sigma = sqrt(1/beta+dot(X,X,1));   % 3.59
    p = exp(((t-y).^2/sigma2+log(2*pi*sigma2))/(-2));
end

% if nargout > 1
%     beta = model.beta;
%     if isfield(model,'V')   % V*V'=inv(S) 3.54
%         U = model.V'*bsxfun(@minus,X,model.xbar);
%         sigma = sqrt(1/beta+dot(U,U,1));   % 3.59
%     else
%         sigma = sqrt(1/beta);
%     end
%     if nargin == 3 && nargout == 3
%         p = exp(logGauss(t,y,sigma));
% %         p = exp(-0.5*(((t-y)./sigma).^2+log(2*pi))-log(sigma));
%     end
% end
