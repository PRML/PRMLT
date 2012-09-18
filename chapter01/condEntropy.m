function z = condEntropy (x, y)
% Compute conditional entropy H(x|y) of two discrete variables x and y.
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
Pxy = nonzeros(Mx'*My/n); %joint distribution of x and y
Hxy = -dot(Pxy,log2(Pxy));

Py = nonzeros(mean(My,1));
Hy = -dot(Py,log2(Py));

% conditional entropy H(x|y)
z = Hxy-Hy;

