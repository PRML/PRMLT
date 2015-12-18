function X = knPcaPred(model, Xt, opt)
% Prediction for kernel PCA
%   model: trained model structure
%   X: d x n testing data
%   t (optional): 1 x n testing response
% Written by Mo Chen (sth4nth@gmail.com).

U = model.U;
L = model.L;

if nargin == 3 && opt.whiten
    Y = ;
end

