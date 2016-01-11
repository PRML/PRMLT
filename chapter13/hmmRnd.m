function [ X, M, A, E, s ] = hmmRnd(d, k, n)
% Generate a data sequence from a hidden Markov model
A = normalize(rand(k,k),2);
E = normalize(rand(k,d),2);
s = normalize(rand(k,1),1);

z = zeros(1,n);
x = zeros(1,n);
z(1) = discreteRnd(s);
x(1) = discreteRnd(E(z(1),:));
for i = 2:n
    z(i) = discreteRnd(A(z(i-1),:));
    x(i) = discreteRnd(E(z(i),:));
end
X = sparse(x,1:n,1,d,n);
M = E*X;
