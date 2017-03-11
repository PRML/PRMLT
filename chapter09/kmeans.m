function [label, m, energy] = kmeans(X, init)
% Perform kmeans clustering.
% Input:
%   X: d x n data matrix
%   init: k number of clusters or label (1 x n vector)
% Output:
%   label: 1 x n cluster label
%   energy: optimization target value
%   model: trained model structure
% Written by Mo Chen (sth4nth@gmail.com).
n = size(X,2);
idx = 1:n;
last = zeros(1,n);
if numel(init)==1
    k = init;
    label = ceil(k*rand(1,n));
elseif numel(init)==n
    label = init;
end
while any(label ~= last)
    [~,~,last(:)] = unique(label);   % remove empty clusters
    E = sparse(idx,last,1);  % transform label into indicator matrix
    m = X*(E./sum(E,1));    % compute centers 
    [val,label] = min(dot(m,m,1)'/2-m'*X,[],1); % assign labels
end
energy = dot(X(:),X(:),1)+2*sum(val);