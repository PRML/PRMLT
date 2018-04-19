function [nodeBel, edgeBel, L] = mrfMf(A, nodePot, edgePot, epoch)
% Mean field for MRF
% Assuming egdePot is symmetric
% Input: 
%   A: n x n adjacent matrix of undirected graph, where value is edge index
%   nodePot: k x n node potential
%   edgePot: k x k x m edge potential
% Output:
%   nodeBel: k x n node belief
%   edgeBel: k x k x m edge belief
% Written by Mo Chen (sth4nth@gmail.com)
if nargin < 4
    epoch = 10;
end
L = -inf(1,epoch+1);
[nodeBel,lnZ] = softmax(nodePot,1);    % initialization    
for iter = 1:epoch
    for i = 1:size(nodePot,2)
        [~,j,e] = find(A(i,:));             % neighbors
        [nodeBel(:,i),lnZ(i)] = softmax(nodePot(:,i)+reshape(edgePot(:,:,e),2,[])*reshape(nodeBel(:,j),[],1));
    end
%     E = dot(nodeBel,nodePot,1);
%     H = -dot(nodeBel,log(nodeBel),1);
%     L(iter+1) = sum(lnZ+E+H)/2;
    L(iter+1) = mrfGibbs(A,nodePot,edgePot,nodeBel);
%     if abs(L(iter+1)-L(iter))/abs(L(iter)) < tol; break; end
end
L = L(1,2:iter+1);

[s,t,e] = find(triu(A));
edgeBel = zeros(size(edgePot));
for l = 1:numel(e)
    edgeBel(:,:,e(l)) = nodeBel(:,s(l))*nodeBel(:,t(l))';
end