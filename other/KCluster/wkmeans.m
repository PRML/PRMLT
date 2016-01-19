function [label, energy, m] = wkmeans(X, init, w)
% Perform weighted k-means clustering.
%   X: d x n data matrix
%   init: k (1 x 1) or label (1 x n, 1<=label(i)<=k) or center (d x k)
%   w: 1 x n weight vector (default w=1, equivalent to kmeans.
% Written by Michael Chen (sth4nth@gmail.com).
%% initialization
if nargin == 2
    w = 1;
end
[d,n] = size(X);
if length(init) == 1
    idx = randsample(n,init);
    m = X(:,idx);
    [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
elseif size(init,1) == 1 && size(init,2) == n
    label = init;
elseif size(init,1) == d
    m = init;
    [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
else
    error('ERROR: init is not valid.');
end
%% main algorithm
last = 0;
while any(label ~= last)
    [u,~,label] = unique(label);   % remove empty clusters
    k = length(u);
    E = sparse(1:n,label,w,n,k,n);
    m = bsxfun(@times,X*E,1./full(sum(E,1)));
    last = label;
    [val,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
end
[~,~,label] = unique(label);   % remove empty clusters
energy = -2*sum(val)+dot(X(:),X(:)); % sum of distances of clusters

% s = energy/(n-k); % variance