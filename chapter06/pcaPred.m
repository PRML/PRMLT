function Y = pcaPred( model, Xt, opt)
% Prediction for PCA: project future data to principal subspace
%   model: trained model structure
%   Xt: d x n testing data
% Written by Mo Chen (sth4nth@gmail.com).
xbar = model.xbar;
U = model.U;
Y = U'*bsxfun(@minus,Xt,xbar);
if nargin == 3 && opt.whiten
    L = model.L;
    Y = bsxfun(@times,Y,1./sqrt(L));
end



