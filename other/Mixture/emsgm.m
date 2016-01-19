function [label, model, llh] = emsgm(X, init)
% EM algorithm for spherical (isotropic) Gaussian mixture model
% Written by Michael Chen (sth4nth@gmail.com).
fprintf('EM for spherical (isotropic) Gaussian mixture: running ... \n');
R = initialization(X, init);
[~,label(1,:)] = max(R,[],2);
R = R(:,unique(label));

[d,n] = size(X);
tol = 1e-6;
maxiter = 500;
llh = -inf(1,maxiter);
converged = false;
t = 1;
X2 = repmat(dot(X,X,1)',1,size(R,2));
while ~converged && t < maxiter
    t = t+1;
    % maximizition step
    nk = sum(R,1);
    w = nk/n;
    mu = bsxfun(@times,X*R,1./nk);
    D = bsxfun(@plus,X2-2*X'*mu,dot(mu,mu,1));
    sigma = dot(D,R,1)./(d*nk);

    % expectation step
    M = bsxfun(@times,D,1./sigma);  % M distance
    c = d*log(2*pi*sigma)/(-2);          % normalization constant
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