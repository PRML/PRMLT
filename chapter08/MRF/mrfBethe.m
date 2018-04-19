function lnZ = mrfBethe(A, nodePot, edgePot, nodeBel, edgeBel)
% Compute Bethe energy
[s,t,e] = find(triu(A));
edgeCor = zeros(size(edgePot));
for l = 1:numel(e)
    edgeCor(:,:,e(l)) = edgeBel(:,:,e(l))./(nodeBel(:,s(l))*nodeBel(:,t(l))');
end
Ex = dot(nodeBel(:),nodePot(:));
Exy = dot(edgeBel(:),edgePot(:));
Hx = -dot(nodeBel(:),log(nodeBel(:)));
Ixy = dot(edgeBel(:),log(edgeCor(:)));
lnZ = Ex+Exy+Hx-Ixy;