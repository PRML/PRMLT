function [label, energy, model] = kmeans(X, init)
%  Perform k-means clustering.
%   X: d x n data matrix
%   k: number of seeds
% Written by Mo Chen (sth4nth@gmail.com).
n = size(X,2);
if numel(init)==1
    k = init;
    label = ceil(k*rand(1,n));
elseif numel(init)==n
    label = init;
    k = max(label);
end
last = 0;
while any(label ~= last)
    E = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix
    m = X*(E*spdiags(1./sum(E,1)',0,k,k));    % compute m of each cluster
    last = label;
    [val,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1); % assign samples to the nearest centers
end
energy = dot(X(:),X(:))-2*sum(val);   % not consist with knKmeans
model.means = m;