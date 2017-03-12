function [label, mu] = litekmeans(X, k)
n = size(X,2);
last = zeros(1,n);
label = ceil(k*rand(1,n));
while any(label ~= last)
    [~,~,last(:)] = unique(label);            % remove empty clusters
    mu = X*normalize(sparse(1:n,last,1),1);    % compute cluster centers 
    [~,label] = min(dot(mu,mu,1)'/2-mu'*X,[],1); % assign sample labels
end
% Perform kmeans clustering.
% Input:
%   X: d x n data matrix
%   k: number of clusters
% Output:
%   label: 1 x n cluster label
%   mu: d x k center of clusters
% Written by Mo Chen (sth4nth@gmail.com).