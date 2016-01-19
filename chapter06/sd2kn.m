function K = sd2kn(D)
% Transform a square distance matrix to a kernel matrix. 
% The data are assumed to be centered, i.e., H=eye(n)-ones(n)/n; K=-(H*D*H)/2;
% Written by Michael Chen (sth4nth@gmail.com).
D = bsxfun(@minus,D,mean(D,1));
D = bsxfun(@minus,D,mean(D,2));
K = (D+D')/(-4);