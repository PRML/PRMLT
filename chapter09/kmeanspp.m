function [label, mu, energy] =  kmeanspp(X, k)
% Perform kmeans clustering.
% Input:
%   X: d x n data matrix
%   k: number of clusters
% Output:
%   label: 1 x n sample labels
%   mu: d x k center of clusters
%   energy: optimization target value
% Written by Mo Chen (sth4nth@gmail.com).
[label, mu, energy] = kmeans(X, kseeds(X,k));

% TBD: label and energy
function [label, mu, energy] = kseeds(X, k)
% kmeans++ seeding
[d,n] = size(X);
v = inf(1,n);
mu = zeros(d,k);
mu(:,1) = X(:,ceil(n*rand));
label = zeros(1,n);
for i = 2:k
    X0 = X-mu(:,i-1);
    [v,label] = min(v,dot(X0,X0,1));
    mu(:,i) = X(:,randp(v));
end
energy = sum(v);

% Done
function idx = randp(p)
% sample one of k by probability
p = cumsum(p);
p = p/p(end);
idx = find(rand<p,1);

% Done
function [label, mu, energy] = kmeans(X, label)
% standard kmeans (Lloyd iteration)
idx = 1:size(X,2);
last = idx;
while any(label ~= last)
    [~,~,last(:)] = unique(label);            % remove empty clusters
    mu = X*normalize(sparse(idx,last,1),1);    % compute cluster centers 
    [val,label] = min(dot(mu,mu,1)'/2-mu'*X,[],1); % assign sample labels
end
energy = dot(X(:),X(:),1)+2*sum(val);