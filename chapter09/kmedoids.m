function [label, energy, index] = kmedoids(X, init)
% Perform k-medoids clustering.
% Input:
%   X: d x n data matrix
%   init: k number of clusters or label (1 x n vector)
% Output:
%   label: 1 x n cluster label
%   energy: optimization target value
%   index: index of medoids
% Written by Mo Chen (sth4nth@gmail.com).
[d,n] = size(X);
if numel(init)==1
    k = init;
    label = ceil(k*rand(1,n));
elseif numel(init)==n
    label = init;
end
X = bsxfun(@minus,X,mean(X,2));             % reduce chance of numerical problems
v = dot(X,X,1);
D = bsxfun(@plus,v,v')-2*(X'*X);            % Euclidean distance matrix
D(sub2ind([d,d],1:d,1:d)) = 0;              % reduce chance of numerical problems
last = 0;
while any(label ~= last)
    [u,~,label(:)] = unique(label);   % remove empty clusters
    [~, index] = min(D*sparse(1:n,label,1,n,numel(u),n),[],1);  % find k medoids
    last = label;
    [val, label] = min(D(index,:),[],1);                % assign labels
end
energy = sum(val);
