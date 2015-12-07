function [ output_args ] = logitPred(model, X, t )
% Prodict the label for binary lgoistic regression model
% model: trained model structure
%   X: d x n testing data
%   t (optional): 1 x n testing label
% Written by Mo Chen (sth4nth@gmail.com).
w = model.w;
w0 = model.w0;
y = w'*X+w0;
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


