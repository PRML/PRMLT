function m = farseeds(X, k)
% Find k farest samples as seeds for initializing clustering.
%   X: d x n data matrix
%   k: number of seeds
% Written by Michael Chen (sth4nth@gmail.com).
d = size(X,1);
m = zeros(d,k);
% idx = ceil(n.*rand);
[~,idx] = max(dot(X,X,1));
m(:,1) = X(:,idx);
D = 0;
for i = 2:k
    Y = bsxfun(@minus,X,m(:,i-1));
    D = D+sqrt(dot(Y,Y,1));
    [~,idx] = max(D);
    m(:,i) = X(:,idx);    
end
