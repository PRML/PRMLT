function [Q, R] = gsog(X)
% Gram-Schmidt orthogonalization
% Written by Mo Chen (sth4nth@gmail.com).
[d,n] = size(X);
m = min(d,n);
R = eye(m,n);
Q = zeros(d,m);
D = zeros(1,m);
for i = 1:m
    R(1:i-1,i) = bsxfun(@times,Q(:,1:i-1),1./D(1:i-1))'*X(:,i);
    Q(:,i) = X(:,i)-Q(:,1:i-1)*R(1:i-1,i);
    D(i) = dot(Q(:,i),Q(:,i));
end
R(:,m+1:n) = bsxfun(@times,Q,1./D)'*X(:,m+1:n);