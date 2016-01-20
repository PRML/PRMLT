function [label, model, llh] = mixGaussGb(X, prior)
% Collapsed Gibbs sampling for (infinite) Gaussian mixture model (a.k.a.
% DPGM)
n = size(X,2);

Theta = prior.theta;
alpha = prior.alpha;
Z = ones(1,n);
maxIter = 1000;
for iter = 1:maxIter
    for i = randperm(n)
        rest = true(1,n);
        rest(i) = false;
        x = X(:,i);
        z = Z(:,i);
        
        Theta{z} = delSample(Theta{z},x);
        logFx = cellfun(x,Theta.Pred);
        logNk = log(sum(Z(:,rest),2));
        logNk(1) = log(alpha);
        logR = logFx+logNk;
        p = exp(logR-logsumexp(logR));
        z = discreteRnd(p);
        Z(:,i) = z;
        Theta{z} = addSample(Theta{z},x);
        index = any(Z,2);
        Z = [zeros(1,n);Z(index)];
        Theta = {prior.theta,Theta{index}};                  % remove empty
    end
end

