function [label, model] = kspheres(X, k)
% Clustering samples into k isotropic Gaussian with different variances.
%   X: d x n data matrix
%   k: number of seeds
% Written by Michael Chen (sth4nth@gmail.com).
[d,n] = size(X);
last = 0;
label = ceil(k*rand(1,n));  % random initialization
while any(label ~= last)
    [u,~,label] = unique(label);   % remove empty clusters
    k = length(u);
    R = sparse(label,1:n,1,k,n,n);
    nk = sum(R,2);
    w = nk/n;
    mu = bsxfun(@times, X*R', 1./nk');
    
    D = sqdistance(mu,X);
    s = dot(D,R,2)./(d*nk);
    
    R = bsxfun(@times,D,1./s);
    R = bsxfun(@plus,R,d*log(2*pi*s))/(-2);
    R = bsxfun(@plus,R,log(w));
    
    last = label;
    [~,label] = max(R,[],1);
end
[~,~,label] = unique(label);   % remove empty clusters
model.mu = mu;
model.sigma = s';
model.weight = w;