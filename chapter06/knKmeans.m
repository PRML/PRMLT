function [label, energy, model] = knKmeans(X, init, kn)
% Perform kernel k-means clustering.
%   K: nxn kernel matrix
%   k: number of cluster
% Reference: Kernel Methods for Pattern Analysis
% by John Shawe-Taylor, Nello Cristianini
% Written by Mo Chen (sth4nth@gmail.com).
n = size(X,2);
if numel(init)==1
    k = init;
    label = ceil(k*rand(1,n));
elseif numel(init)==n
    label = init;
    k = max(label);
end
K = kn(X,X);
last = 0;
while any(label ~= last)
    E = sparse(label,1:n,1,k,n,n);
    E = spdiags(1./sum(E,2),0,k,k)*E;
    T = E*K;
    last = label;
    [val, label] = max(bsxfun(@minus,T,diag(T*E')/2),[],1);
%     [val, label] = max(bsxfun(@minus,2*T,dot(T,E,2)),[],1);
end
energy = trace(K)-2*sum(val); 
if nargout == 3
    model.X = X;
    model.label = label;
    model.kn = kn;
end
