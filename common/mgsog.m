function [Q, R] = mgsog(X)
% Modified Gram-Schmidt orthogonalization
% Written by Mo Chen (sth4nth@gmail.com).
[d,n] = size(X);
m = min(d,n);
R = eye(m,n);
Q = zeros(d,m);
D = zeros(1,m);
for i = 1:m
    v = X(:,i);
    for j = 1:i-1
        R(j,i) = Q(:,j)'*v/D(j);
        v = v-R(j,i)*Q(:,j);
    end
    Q(:,i) = v;
    D(i) = dot(Q(:,i),Q(:,i));
end
R(:,m+1:n) = bsxfun(@times,Q,1./D)'*X(:,m+1:n);