function [label, energy, model] = kmeans(X, init)
% Perform k-means clustering.
% Input:
%   X: d x n data matrix
%   init: k number of clusters or label (1 x n vector)
% Output:
%   label: 1 x n cluster label
%   energy: optimization target value
%   model: trained model structure
% Written by Mo Chen (sth4nth@gmail.com).
n = size(X,2);
if numel(init)==1
    k = init;
    label = ceil(k*rand(1,n));
elseif numel(init)==n
    label = init;
end
last = 0;
while any(label ~= last)
    [u,~,label(:)] = unique(label);   % remove empty clusters
    k = numel(u);
    E = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix
    m = X*(E*spdiags(1./sum(E,1)',0,k,k));    % compute centers 
    last = label;
    [val,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1); % assign labels
end
energy = dot(X(:),X(:))-2*sum(val); 
model.means = m;