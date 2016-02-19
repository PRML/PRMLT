function z = entropy(x)
% Compute entropy z=H(x) of a discrete variable x.
% Input:
%   x: a integer vectors  
% Output:
%   z: entropy z=H(x)
% Written by Mo Chen (sth4nth@gmail.com).
n = numel(x);
[u,~,x] = unique(x);
k = numel(u);
idx = 1:n;
Mx = sparse(idx,x,1,n,k,n);
Px = nonzeros(mean(Mx,1));
Hx = -dot(Px,log2(Px));
z = max(0,Hx);