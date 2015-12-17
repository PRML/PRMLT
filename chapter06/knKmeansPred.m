function label = knKmeansPred(model, Xt)
% Prediction for kernel kmeans clusterng
%   model: trained model structure
%   Xt: d x n testing data
% Written by Mo Chen (sth4nth@gmail.com).
X = model.X;
t = model.label;
kn = model.kn;

n = size(X,2);
k = max(t);
E = sparse(t,1:n,1,k,n,n);
E = bsxfun(@times,E,1./sum(E,2));
Z = bsxfun(@plus,-2*E*kn(X,Xt),diag(E*kn(X,X)*E'));
[~, label] = min(Z,[],1);
