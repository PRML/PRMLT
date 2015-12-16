function [label, energy, model] = knKmeans(X, k, kn)
% Perform kernel k-means clustering.
%   K: nxn kernel matrix
%   k: number of cluster
% Reference: Kernel Methods for Pattern Analysis
% by John Shawe-Taylor, Nello Cristianini
% Written by Mo Chen (sth4nth@gmail.com).
K = kn(X,X);
n = size(X,2);
label = ceil(k*rand(1,n));
last = 0;
while any(label ~= last)
    E = sparse(label,1:n,1,k,n,n);
    E = bsxfun(@times,E,1./sum(E,2));
    T = E*K;
    Z = repmat(diag(T*E'),1,n)-2*T;
    last = label;
    [val, label] = min(Z,[],1);
end
energy = sum(val)+trace(K);
if nargout == 3
    model.X = X;
    model.label = label;
    model.kn = kn;
end