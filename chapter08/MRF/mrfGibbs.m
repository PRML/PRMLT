function lnZ = mrfGibbs(A, nodePot, edgePot, nodeBel)
% Compute Gibbs energy
[s,t,e] = find(triu(A));
edgeBel = zeros(size(edgePot));
for l = 1:numel(e)
    edgeBel(:,:,e(l)) = nodeBel(:,s(l))*nodeBel(:,t(l))';
end
Ex = dot(nodeBel(:),nodePot(:));
Exy = dot(edgeBel(:),edgePot(:));
Hx = -dot(nodeBel(:),log(nodeBel(:)));
lnZ = Ex+Exy+Hx;