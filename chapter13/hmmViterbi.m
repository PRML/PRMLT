function [z, v] = hmmViterbi (x, s, P, E)
% x: 1xn observation sequence vector
% s: starting probability (prior)
% P: transition probability matrix
% E: Emmission probability matrix
n = length(x);
k = length(s);
Z = zeros(k,n);
P = log(P);
E = log(E);
Z(:,1) = 1:k;
v = log(s(:))+E(:,x(1));
for t = 2:n
    [v,idx] = max(bsxfun(@plus,v,P),[],1);
    v = v(:)+E(:,x(t));
    Z = Z(idx,:);
    Z(:,t) = 1:k;
end
[v,idx] = max(v);
z = Z(idx,:);
v = exp(v);
