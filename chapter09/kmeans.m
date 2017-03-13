function [label, mu, energy] = kmeans(X, init)
% Perform kmeans clustering.
% Input:
%   X: d x n data matrix
%   init: k number of clusters or label (1 x n vector)
% Output:
%   label: 1 x n sample labels
%   mu: d x k center of clusters
%   energy: optimization target value
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
    mu = X*normalize(sparse(idx,last,1),1);    % compute centers 
    [val,label] = min(dot(mu,mu,1)'/2-mu'*X,[],1); % assign labels
end
energy = dot(X(:),X(:),1)+2*sum(val);