function X = mgs(X)
% Modified Gram-Schmidt orthogonalization
for i = 1:size(X,2)
    v = X(:,i)-X(:,1:i-1)*(X(:,1:i-1)'*X(:,i));
    X(:,i) = v/sqrt(dot(v,v));
end