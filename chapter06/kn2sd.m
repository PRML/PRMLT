function D = kn2sd(K)
% Transform a kernel matrix (or inner product matrix) to a square distance matrix
% Written by Michael Chen (sth4nth@gmail.com).
d = diag(K);
D = -2*K+bsxfun(@plus,d,d');
