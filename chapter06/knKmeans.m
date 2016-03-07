function [label, energy, model] = knKmeans(X, init, kn)
% Perform kernel k-means clustering.
% Input:
%   K: n x n kernel matrix
%   init: either number of clusters (k) or initial label (1xn)
% Output:
%   label: 1 x n clustering result label
%   energy: optimization target value
%   model: trained model structure
% Reference: Kernel Methods for Pattern Analysis
% by John Shawe-Taylor, Nello Cristianini
% Written by Mo Chen (sth4nth@gmail.com).
n = size(X,2);
if numel(init)==1
    k = init;
    label = ceil(k*rand(1,n));
elseif numel(init)==n
    label = init;
end
if nargin < 3
    kn = @knGauss;
end
K = kn(X,X);
last = 0;
while any(label ~= last)
    [u,~,label(:)] = unique(label);   % remove empty clusters
    k = numel(u);
    E = sparse(label,1:n,1,k,n,n);
    E = spdiags(1./sum(E,2),0,k,k)*E;
    T = E*K;
    last = label;
    [val, label] = max(bsxfun(@minus,T,diag(T*E')/2),[],1);
end
energy = trace(K)-2*sum(val); 
if nargout == 3
    model.X = X;
    model.label = label;
    model.kn = kn;
end
