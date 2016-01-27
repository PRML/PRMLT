function [argmax, prob] = hmmViterbi_(M, A, s)
% Implmentation function of Viterbi algorithm. (not supposed to be called
% directly)
% M: data matrix
% A: transition probability matrix
% s: starting probability (prior)
% Written by Mo Chen (sth4nth@gmail.com).
[k,n] = size(M);
Z = zeros(k,n);
A = log(A);
M = log(M);
Z(:,1) = 1:k;
v = log(s(:))+M(:,1);
for t = 2:n
    [v,idx] = max(bsxfun(@plus,A,v),[],1);    % 13.68
    v = v(:)+M(:,t);
    Z = Z(idx,:);
    Z(:,t) = 1:k;
end
[v,idx] = max(v);
argmax = Z(idx,:);
prob = exp(v);
