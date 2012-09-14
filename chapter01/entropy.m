function z = entropy(x)
% Compute entropy H(x) of a discrete variable x.
% Written by Mo Chen (mochen80@gmail.com).
n = numel(x);
x = reshape(x,1,n);
[u,~,label] = unique(x);
p = full(mean(sparse(1:n,label,1,n,numel(u),n),1));
z = -dot(p,log2(p+eps));
