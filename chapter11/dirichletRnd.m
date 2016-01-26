function x = dirichletRnd(a, m)
% Sampling from a Dirichlet distribution.
%   a: k dimensional vector
%   m: k dimensional mean vector
% Written by Mo Chen (sth4nth@gmail.com).
if nargin == 2
    a = a*m;
end
x = gamrnd(a,1);
x = x/sum(x);
