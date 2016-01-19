function [label, model, L] = vbdgm(X, init, prior)
% Perform variational Bayesian inference for Gaussian mixture.
%   X: d x n data matrix
%   init: k (1 x 1) or label (1 x n, 1<=label(i)<=k) or center (d x k)
% Reference: Pattern Recognition and Machine Learning by Christopher M. Bishop (P.474)
% Written by Michael Chen (sth4nth@gmail.com).

fprintf('Variational Bayesian Gaussian mixture: running ... \n');
[d,n] = size(X);
if nargin < 3
    prior.alpha = 1;
    prior.kappa = 1;
    prior.m = mean(X,2);
    prior.nu = d+1;
    prior.M = eye(d);   % M = inv(W)
end
tol = 1e-10;
maxiter = 1000;
L = -inf(1,maxiter);
converged = false;
t = 1;

model.R = initialization(X,init);

while  ~converged && t < maxiter
    t = t+1;
    model = qDirichlet(model, prior);
    model = qGaussianWishart(X, model, prior);
    model = qMultinomial(X, model);
    L(t) = bound(model,prior)/n;
    converged = abs(L(t)-L(t-1)) < tol*abs(L(t));
end
L = L(2:t);
label = zeros(1,n);
[~,label(:)] = max(model.R,[],2);
[~,~,label] = unique(label);
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


% Done
function model = qDirichlet(model, prior)
alpha0 = prior.alpha;
R = model.R;

nk = sum(R,1); % 10.51
alpha = alpha0+nk; % 10.58

model.alpha = alpha;
 
% Done
function model = qGaussianWishart(X, model, prior)
kappa0 = prior.kappa;
m0 = prior.m;
nu0 = prior.nu;
M0 = prior.M;
R = model.R;

nk = sum(R,1); % 10.51
nxbar = X*R;

kappa = kappa0+nk; % 10.60
m = bsxfun(@times,bsxfun(@plus,kappa0*m0,nxbar),1./kappa); % 10.61
nu = nu0+nk; % 10.63


[d,k] = size(m);
M = zeros(d,d,k); 
sqrtR = sqrt(R);

xbar = bsxfun(@times,nxbar,1./nk); % 10.52
xbarm0 = bsxfun(@minus,xbar,m0);
w = (kappa0*nk./(kappa0+nk));
for i = 1:k
    Xs = bsxfun(@times,bsxfun(@minus,X,xbar(:,i)),sqrtR(:,i)');
    xbarm0i = xbarm0(:,i);
    M(:,:,i) = M0+Xs*Xs'+w(i)*(xbarm0i*xbarm0i'); % 10.62
end

model.kappa = kappa;
model.m = m;
model.nu = nu;
model.M = M; % Whishart: M = inv(W)


% Done
function model = qMultinomial(X, model)
alpha = model.alpha; % Dirichlet
kappa = model.kappa;   % Gaussian
m = model.m;         % Gasusian
nu = model.nu;         % Whishart
M = model.M;         % Whishart: inv(W) = V'*V

n = size(X,2);
[d,k] = size(m);

logW = zeros(1,k);
EQ = zeros(n,k);
for i = 1:k
    U = chol(M(:,:,i));
    logW(i) = -2*sum(log(diag(U)));      
    Q = (U'\bsxfun(@minus,X,m(:,i)));
    EQ(:,i) = d/kappa(i)+nu(i)*dot(Q,Q,1);    % 10.64
end

ElogLambda = sum(psi(0,bsxfun(@minus,nu+1,(1:d)')/2),1)+d*log(2)+logW; % 10.65
Elogpi = psi(0,alpha)-psi(0,sum(alpha)); % 10.66

logRho = (bsxfun(@minus,EQ,2*Elogpi+ElogLambda-d*log(2*pi)))/(-2); % 10.46
logR = bsxfun(@minus,logRho,logsumexp(logRho,2)); % 10.49
R = exp(logR);

model.logR = logR;
model.R = R;

function L = bound(model, prior)
alpha0 = prior.alpha;
alpha = model.alpha;
R = model.R;
logR = model.logR;

nk = sum(R,1); % 10.51
k = size(R,2);
Elogpi = psi(0,alpha)-psi(0,sum(alpha));

Epz = dot(nk,Elogpi);
Eqz = dot(R(:),logR(:));
logCalpha0 = gammaln(k*alpha0)-k*gammaln(alpha0);
Eppi = logCalpha0+(alpha0-1)*sum(Elogpi);
logCalpha = gammaln(sum(alpha))-sum(gammaln(alpha));
Eqpi = logCalpha+dot(alpha-1,Elogpi);

L = Epz-Eqz+Eppi-Eqpi;