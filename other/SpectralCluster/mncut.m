function label = mncut(W, c, m)
% Multiway normailized cut
% W: symetric affinity matrix
% c: number of clusters
% m: {1,2,3,4} method for discretization
if nargin < 3
    m = 1;
end
[L,d] = laplacian(W,'n');
V = symeig(L,c)';
label = discretize(V,d,m);