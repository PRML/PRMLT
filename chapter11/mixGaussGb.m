function [label, model, llh] = mixGaussGb(X, init)
% Collapsed Gibbs sampling for (infinite) Gaussian mixture model (a.k.a.
% DPGM)


[d,n] = size(X);
maxIter = 1000;
for iter = 1:maxIter
    for i = randperm(n)
        
    end
end