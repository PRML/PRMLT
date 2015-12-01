function x = discreternd(p, n)
% Sampling from a discrete distribution (multinomial).
% Written by Mo Chen (sth4nth@gmail.com).
if nargin == 1
    n = 1;
end
r = rand(1,n);
p = cumsum(p(:));
[~,x] = histc(r,[0;p/p(end)]);
