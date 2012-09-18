function z = relatEntropy (x, y)
% Compute relative entropy (a.k.a KL divergence) KL(p(x)||p(y)) of two discrete variables x and y.
% Written by Mo Chen (mochen80@gmail.com).    
assert(numel(x) == numel(y));
n = numel(x);
x = reshape(x,1,n);
y = reshape(y,1,n);

l = min(min(x),min(y));
x = x-l+1;
y = y-l+1;
k = max(max(x),max(y));

idx = 1:n;
Mx = sparse(idx,x,1,n,k,n);
My = sparse(idx,y,1,n,k,n);
Px = nonzeros(mean(Mx,1));
Py = nonzeros(mean(My,1));

z = -dot(Px,log2(Py)-log2(Px));

