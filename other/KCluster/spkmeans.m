function [label, m, energy] = spkmeans(X, init)
% Perform spherical k-means clustering.
%   X: d x n data matrix
%   init: k (1 x 1) or label (1 x n, 1<=label(i)<=k) or center (d x k)
% Reference: Clustering on the Unit Hypersphere using Von Mises-Fisher Distributions.
% by A. Banerjee, I. Dhillon, J. Ghosh and S. Sra.
% Written by Michael Chen (sth4nth@gmail.com).
%% initialization
[d,n] = size(X);
X = normalize(X);

if length(init) == 1
    idx = randsample(n,init);
    m = X(:,idx);
    [~,label] = max(m'*X,[],1);
elseif size(init,1) == 1 && size(init,2) == n
    label = init;
elseif size(init,1) == d
    m = normalize(init);
    [~,label] = max(m'*X,[],1);
else
    error('ERROR: init is not valid.');
end
%% main algorithm: final version 
last = 0;
while any(label ~= last)
    [u,~,label] = unique(label);   % remove empty clusters
    k = length(u);
    E = sparse(1:n,label,1,n,k,n);
    m = normalize(X*E);
    last = label;
    [val,label] = max(m'*X,[],1);
end
[~,~,label] = unique(label);   % remove empty clusters
energy = sum(val);