function [label, mu] = litekmeans(X, label)
idx = 1:size(X,2);
last = idx;
while any(label ~= last)
    [~,~,last(:)] = unique(label);            % remove empty clusters
    mu = X*normalize(sparse(idx,last,1),1);    % compute cluster centers 
    [~,label] = min(dot(mu,mu,1)'/2-mu'*X,[],1); % assign sample labels
end
% Perform kmeans clustering.
% Input:
%   X: d x n data matrix
%   label: initial sample labels
% Output:
%   label: 1 x n sample label
%   mu: d x k center of clusters
% Written by Mo Chen (sth4nth@gmail.com).