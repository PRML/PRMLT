function [argmax, prob] = hmmViterbi (M, A, s)
% x: 1xn observation sequence vector
% s: starting probability (prior)
% A: transition probability matrix
% E: Emmission probability matrix
% Written by Mo Chen (sth4nth@gmail.com).
[k,n] = size(M);
Z = zeros(k,n);
A = log(A);
M = log(M);
Z(:,1) = 1:k;
v = log(s(:))+M(:,1);
for t = 2:n
    [v,idx] = max(bsxfun(@plus,A,v),[],1);
    v = v(:)+M(:,t);
    Z = Z(idx,:);
    Z(:,t) = 1:k;
end
[v,idx] = max(v);
argmax = Z(idx,:);
prob = exp(v);
