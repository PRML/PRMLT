function [Q, R] = mgs(X)
% Modified Gram-Schmidt (numerical stable version of Gram-Schmidt orthogonalization algorithm) 
% which produces the same result as [Q,R]=qr(X,0)
[d,n] = size(X);
m = min(d,n);
R = zeros(m,n);
Q = zeros(d,m);
for i = 1:m
    R(1:i-1,i) = Q(:,1:i-1)'*X(:,i);
    Q(:,i) = X(:,i)-Q(:,1:i-1)*R(1:i-1,i);
    R(i,i) = norm(Q(:,i));
    Q(:,i) = Q(:,i)/R(i,i);
end
R(:,m+1:n) = Q'*X(:,m+1:n);