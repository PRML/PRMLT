function [label, energy] = knKmeansPred(model, Xt)
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
Z = bsxfun(@minus,E*kn(X,Xt),diag(E*kn(X,X)*E')/2);
[val, label] = max(Z,[],1);
energy = sum(kn(Xt))-2*sum(val);
