function Y = pcaPred( model, X, whiten)
% Prediction for PCA: project future data to principal subspace
%   model: trained model structure
%   X: d x n testing data
% Written by Mo Chen (sth4nth@gmail.com).
mu = model.mu;
U = model.U;
Y = U'*bsxfun(@minus,X,mu);
if nargin == 3 && whiten 
    L = model.L;
    Y = bsxfun(@times,Y,1./sqrt(L));
end



