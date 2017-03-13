function mu = kseeds(X, k)
% Perform kmeans++ seeding
% Input:
%   X: d x n data matrix
%   k: number of seeds
% Output:
%   label: 1 x n sample labels
%   mu: d x k seeds
%   energy: kmeans target value
% Written by Mo Chen (sth4nth@gmail.com).
n = size(X,2);
D = inf(1,n);
mu = X(:,ceil(n*rand));
for i = 2:k
    D = min(D,sum((X-mu(:,i-1)).^2,1));
    mu(:,i) = X(:,randp(D));
end
