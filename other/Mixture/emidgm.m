function [label, model, llh] = emidgm(X, init)
% EM algorithm for independent Gaussian mixture model
% Written by Michael Chen (sth4nth@gmail.com).
fprintf('EM for independent Gaussian mixture: running ... \n');
R = initialization(X, init);
[~,label(1,:)] = max(R,[],2);
R = R(:,unique(label));

[d,n] = size(X);
tol = 1e-6;
maxiter = 500;
llh = -inf(1,maxiter);
converged = false;
t = 1;
X2 = X.^2;
while ~converged && t < maxiter
    t = t+1;
    % maximizition step
    nk = sum(R,1);
    w = nk/n;
    R = bsxfun(@times,R,1./nk);
    mu = X*R;
    mu2 = mu.*mu;
    sigma = X2*R-mu2;

    % expectation step
    lambda = 1./sigma;
    M = bsxfun(@plus,X2'*lambda-2*X'*(mu.*lambda),dot(mu2,lambda,1)); % M distance
    c = (d*log(2*pi)+sum(log(sigma),1))/(-2); % normalization constant

    logRho = bsxfun(@plus,M/(-2),c+log(w));
    T = logsumexp(logRho,2);
    logR = bsxfun(@minus,logRho,T);
    R = exp(logR);
    llh(t) = sum(T)/n; % loglikelihood
    
    [~,label(:)] = max(R,[],2);
    u = unique(label);   % non-empty components
    if size(R,2) ~= size(u,2)
        R = R(:,u);   % remove empty components
    else
        converged = llh(t)-llh(t-1) < tol*abs(llh(t));
    end
end
model.w = w;
model.mu = mu;
model.sigma = sigma;
llh = llh(2:t);
if converged
    fprintf('Converged in %d steps.\n',t-1);
else
    fprintf('Not converged in %d steps.\n',maxiter);
end

function R = initialization(X, init)
[d,n] = size(X);
if length(init) == 1  % random initialization
    k = init;
    idx = randsample(n,k);
    m = X(:,idx);
    [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
    [u,~,label] = unique(label);
    while k ~= length(u)
        idx = randsample(n,k);
        m = X(:,idx);
        [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
        [u,~,label] = unique(label);
    end
    R = full(sparse(1:n,label,1,n,k,n));
elseif size(init,1) == 1 && size(init,2) == n  % initialize with labels
    label = init;
    k = max(label);
    R = full(sparse(1:n,label,1,n,k,n));
elseif size(init,1) == d  %initialize with only centers
    k = size(init,2);
    m = init;
    [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
    R = full(sparse(1:n,label,1,n,k,n));
else
    error('ERROR: init is not valid.');
end
