function [L, d] = laplacian(W, m)
% Compute (normalized) Laplacian matrix from an affinity matrix of an undirected graph.
% input:
%   W: a symmetric adjacent matrix of a undirected graph
%   m: m == 'u' construct unnormalized Laplacian L=D-W
%      m == 'n' construct nomalized Laplacian L=I-D^(-1/2)*W*D^(-1/2)
% Written by Michael Chen (sth4nth@gmail.com).
if nargin == 1
    m = 'u';
end

n = size(W,2);
d = sum(W,1);
if issparse(W)
    switch m
        case 'u'
                L = spdiags(d(:),0,n,n)-W;
        case 'n'
                r = spdiags(sqrt(1./d(:)),0,n,n);
                L = speye(n)-r*W*r;
                L = (L+L')/2;
        otherwise
            error('The parameter is not supported.');
    end
    d = full(d);
else
    switch m
        case 'u'
                L = diag(d)-W;
        case 'n'
                r = sqrt(1./d);
                L = eye(n)-(r'*r).*W;
        otherwise
            error('The parameter is not supported.');
    end
end