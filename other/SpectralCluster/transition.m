function P = transition(W)
% Compute a transition matrix from an affinity matrix.
% Written by Michael Chen (sth4nth@gmail.com).
if issparse(W)
    P = spdiags(1./sum(W,2),0,n,n)*W;
else
    P = bsxfun(@times,W,1./sum(W,2));
end


