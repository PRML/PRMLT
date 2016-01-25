function [label, model, llh] = mixGaussGb(X, prior)
% Collapsed Gibbs sampling for (infinite) Gaussian mixture model (a.k.a.
% DPGM)
[d,n] = size(X);
% parmaters of Gaussian-Wishart prior
if nargin == 1
    kappa0 = 1;
    m0 = zeros(d,1);
    nu0 = d;
    S0 = eye(d);
    alpha0 = 1;
else
    kappa0 = prior.kappa;
    m0 = prior.m;
    nu0 = prior.nu;
    S0 = prior.S;
    alpha0 = prior.alpha;
end

[Theta,Z] = init(X,GaussWishart(kappa0,m0,nu0,S0));
maxIter = 50;
for iter = 1:maxIter
    for i = randperm(n)
        x = X(:,i);
        z = Z(:,i);
        try
        Theta{z} = Theta{z}.delSample(x);
        catch
            error('error!\n');
        end
        logPk = cellfun(@(theta) theta.logPredPdf(x),Theta)';
        logNk = log(sum(Z(:,~id(i,n)),2));
        logNk(1) = log(alpha0);
        logR = logPk+logNk;
        p = exp(logR-logsumexp(logR));
        z = id(discreteRnd(p),numel(p));
        Z(:,i) = z;
        Theta{z} = Theta{z}.addSample(x);
        ne = any(Z,2);   % non-empty
        Z = [false(1,n);Z(ne,:)];
        gw = GaussWishart(kappa0,m0,nu0,S0);
        Theta = {gw,Theta{ne}};                  % remove empty
    end
end
model = Theta{2:end};
label = max(Z(2:end,:),[],1);

function [Theta,Z] = init(X, theta)
n = size(X,2);
for i = randperm(n)
    x = X(:,i);
end


function indicator = id(i, n)
indicator = false(n,1);
indicator(i) = true;
