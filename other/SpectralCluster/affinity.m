function W = affinity(X, sigma, k)
% Construct the affinity matrix of connected undirected graph.
% Wij=exp(-|xi-xj|^2/(2*Sigma))
% Written by Michael Chen (sth4nth@gmail.com).
X = bsxfun(@minus,X,mean(X,2));
S = dot(X,X,1);
if nargin < 3 
    k = 0;
end
if nargin < 2
    sigma = mean(S);
end

n = size(X,2);
D = (-2)*(X'*X)+bsxfun(@plus,S,S');

if k == 0
    W = exp(D/((-2)*sigma));
    W(sub2ind([n,n],1:n,1:n)) = 0; % remove diagonal
else
    [ND, NI] = sort(D);
    ND = ND(2:k+1,:);
    NI = NI(2:k+1,:);
    XI = repmat(1:n,k,1);
    W = sparse(XI(:),NI(:),exp(ND(:)/((-2)*sigma)),n,n);
    W = max(W,W');                  % force symmetry (not necessary for digraph)
end




