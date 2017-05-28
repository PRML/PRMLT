function [nodeBel, edgeBel] = expProp0(A, nodePot, edgePot, epoch)
% Expectation propagation for MRF, calculation in log scale
% Assuming egdePot is symmetric
% Another implementation with precompute nodeBel and update during iterations
% Input: 
%   A: n x n adjacent matrix of undirected graph, where value is edge index
%   nodePot: k x n node potential
%   edgePot: k x k x m edge potential
% Output:
%   nodeBel: k x n node belief
%   edgeBel: k x k x m edge belief
%   L: variational lower bound (Bethe energy)
% Written by Mo Chen (sth4nth@gmail.com)
tol = 0;
if nargin < 4
    epoch = 10;
    tol = 1e-4;
end
k = size(nodePot,1);
m = size(edgePot,3);

[s,t,e] = find(tril(A));
mu = zeros(k,2*m)-log(k);    
nodeBel = -nodePot-logsumexp(-nodePot,1);
for iter = 1:epoch
    mu0 = mu;
    for l = 1:m
        i = s(l);
        j = t(l);
        eij = e(l);
        eji = eij+m;
        ep = edgePot(:,:,eij);

        nodeBel(:,j) = nodeBel(:,j)-mu(:,eij);
        mut = logsumexp(-ep+(nodeBel(:,i)-mu(:,eji)),1);
        mu(:,eij) = mut-logsumexp(mut);
        nb = nodeBel(:,j)+mu(:,eij);
        nodeBel(:,j) = nb-logsumexp(nb);
        
        nodeBel(:,i) = nodeBel(:,i)-mu(:,eji);
        mut = logsumexp(-ep+(nodeBel(:,j)-mu(:,eij)),1);
        mu(:,eji) = mut-logsumexp(mut);
        nb = nodeBel(:,i)+mu(:,eji);
        nodeBel(:,i) = nb-logsumexp(nb);
    end
    if max(abs(mu(:)-mu0(:))) < tol; break; end
end

edgeBel = zeros(k,k,m);
for l = 1:m
    eij = e(l);
    eji = eij+m;
    ep = edgePot(:,:,eij);
    nbt = nodeBel(:,t(l))-mu(:,eij);
    nbs = nodeBel(:,s(l))-mu(:,eji);
    eb = (nbt+nbs')-ep;
    edgeBel(:,:,eij) = eb-logsumexp(eb(:));
end
nodeBel = exp(nodeBel);
edgeBel = exp(edgeBel);