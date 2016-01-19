function label = sc(X, k, opt)
% Perform multiclass spectral clustering (normalized cut).
% Written by Michael Chen (sth4nth@gmail.com).
if nargin < 3
    sigma = 1;
    nnn = 0;
    m = 1;
else
    sigma = fieldvalue(opt,'sigma',1);
    nnn = fieldvalue(opt,'nnn',0);   % number of nearest neighbors
    m = fieldvalue(opt,'method',1);
end

W = affinity(standardize(X),sigma,nnn);
label = mncut(W,k,m);