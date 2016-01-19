function m = softseeds(X, k)
% Find samples as seeds for initializing clustering using kmeans++ algorithm.
%   X: d x n data matrix
%   k: number of seeds
% Reference: k-means++: the advantages of careful seeding. 
% by David Arthur and Sergei Vassilvitskii
% Written by Michael Chen (sth4nth@gmail.com).
[d,n] = size(X);
m = zeros(d,k);
v = inf(1,n);
m(:,1) = X(:,ceil(n*rand));
for i = 2:k
    Y = bsxfun(@minus,X,m(:,i-1));
    v = cumsum(min(v,dot(Y,Y,1)));
    m(:,i) = X(:,find(rand < v/v(end),1));
end