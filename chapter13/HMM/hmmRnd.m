function [x, model] = hmmRnd(d, k, n)
% Generate a data sequence from a hidden Markov model
A = normalize(rand(k,k),2);
E = normalize(rand(k,d),2);
s = normalize(rand(k,1),1);

x = zeros(1,n);
z = discreteRnd(s);
x(1) = discreteRnd(E(z,:));
for i = 2:n
    z = discreteRnd(A(z,:));
    x(i) = discreteRnd(E(z,:));
end

model.A = A;
model.E = E;
model.s = s;