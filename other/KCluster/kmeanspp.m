function [label, energy] =  kmeanspp(X, k)
% X: d x n data matrix
% k: number of seeds
% reference: k-means++: the advantages of careful seeding. David Arthur and Sergei Vassilvitskii
% Written by Michael Chen (sth4nth@gmail.com).
m = seeds(X,k);
[label, energy] = kmeans(X, m);

function m = seeds(X, k)
[d,n] = size(X);
m = zeros(d,k);
v = inf(1,n);
m(:,1) = X(:,ceil(n*rand));
for i = 2:k
    Y = bsxfun(@minus,X,m(:,i-1));
    v = cumsum(min(v,dot(Y,Y,1)));
    m(:,i) = X(:,find(rand < v/v(end),1));
end

function [label, energy] = kmeans(X, m)
n = size(X,2);
last = 0;
[~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
while any(label ~= last)
    [u,~,label] = unique(label);   % remove empty clusters
    k = length(u);
    E = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix
    m = X*(E*spdiags(1./sum(E,1)',0,k,k));    % compute m of each cluster
    last = label;
    [value,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1); % assign samples to the nearest centers
end
[~,~,label] = unique(label);   % remove empty clusters
energy = -2*sum(value)+dot(X(:),X(:)); 