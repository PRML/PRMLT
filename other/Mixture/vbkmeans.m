function [label, model, bound] = vbkmeans(X, init, prior)
% Perform variational Bayesian inference for isotropic Gaussian mixture.
%   X: d x n data matrix
%   init: k (1 x 1) or label (1 x n, 1<=label(i)<=k) or center (d x k)
% Written by Michael Chen (sth4nth@gmail.com).


fprintf('Variantional Bayeisn Kmeans: running ... \n');
[d,n] = size(X);
if nargin < 3
    prior.alpha = 1;        % noninformative setting of Dirichet prior 
    prior.kappa = 1;        % noninformative setting of Gassian prior of Gaussian mean ?
    prior.m = mean(X,2);    % when prior.kappa = 0 it doesnt matter how to set this
    prior.nu = 1;           % noninformative setting of 1d Wishart
    prior.tau = 1;          % noninformative setting of 1d Wishart
end
model.nu = prior.nu+d*n;

R = initialization(X,init);

tol = 1e-8;
maxiter = 5000;
bound = -inf(1,maxiter);
converged = false;
t = 1;

model.R = R;
model = vbmaximization(X,model,prior);  
while  ~converged && t < maxiter
    t = t+1;
    model = vbexpection(X,model);
    model = vbmaximization(X,model,prior);      
    bound(t) = vbound(X,model,prior)/n;
    converged = abs(bound(t)-bound(t-1)) < tol*abs(bound(t));
end
bound = bound(2:t);
label = zeros(1,n);
[~,label(:)] = max(model.R,[],2);
[~,~,label] = unique(label);

if converged
    fprintf('Converged in %d steps.\n',t-1);
else
    fprintf('Not converged in %d steps.\n',maxiter);
end

% Done.

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

% Done.
% update latent variables
function model = vbexpection(X, model)
alpha = model.alpha;
kappa = model.kappa;
m = model.m;
nu = model.nu;
tau = model.tau;

d = size(X,1);

logw = psi(0,alpha)-psi(0,sum(alpha));
loglambda = psi(0,nu/2)-log(tau/2);

M = bsxfun(@plus,sqdistance(X,m)*nu/tau,d./kappa);
c = d*(loglambda-log(2*pi))/2;
logRho = bsxfun(@plus,M/(-2),logw+c);

% [~,idx] = max(logR,[],2);
% logR = logR(:,unique(idx));   % remove empty components!!!

logR = bsxfun(@minus,logRho,logsumexp(logRho,2));
R = exp(logR);

model.logR = logR;
model.R = R;

% Done.
% update the parameters
function model = vbmaximization(X, model, prior)
alpha0 = prior.alpha;        % Dirichet prior 
kappa0 = prior.kappa;    % piror of Gaussian mean
m0 = prior.m;            % piror of Gaussian mean
tau0 = prior.tau;        % 1d Wishart

R = model.R;

% Dirichlet
nk = sum(R,1);
alpha = alpha0+nk; 
% Gaussian
kappa = kappa0+nk;
xbar = bsxfun(@times,X*R,1./nk);
m = bsxfun(@times,bsxfun(@plus,kappa0*m0,bsxfun(@times,xbar,nk)),1./kappa);
% 1d Wishart
Q = sqdistance(X,xbar);
tau = tau0+dot(Q(:),R(:))+dot((kappa0*nk./(kappa0+nk)),sqdistance(m0,xbar));

model.alpha = alpha;
model.kappa = kappa;
model.m = m;
model.tau = tau;


function bound = vbound(X, model, prior)
alpha0 = prior.alpha;        % Dirichet prior 
kappa0 = prior.kappa;    % piror of Gaussian mean
m0 = prior.m;            % piror of Gaussian mean
nu0 = prior.nu;          % 1d Wishart
tau0 = prior.tau;        % 1d Wishart

alpha = model.alpha;
kappa = model.kappa;
m = model.m;
nu = model.nu;
tau = model.tau;

logR = model.logR;
R = model.R;

[d,k] = size(m);

nk = sum(R,1);
logw = psi(0,alpha)-psi(0,sum(alpha));

Epz = nk*logw';
Eqz = R(:)'*logR(:);
logCalpha0 = gammaln(k*alpha0)-k*gammaln(alpha0);
Epw = logCalpha0+(alpha0-1)*sum(logw);
logCalpha = gammaln(sum(alpha))-sum(gammaln(alpha));
Eqw = sum((alpha-1).*logw)+logCalpha;

loglambda = psi(0,nu/2)-log(tau/2);
aib = nu./tau;
Q = bsxfun(@minus,m,m0);
Epmu = 0.5*(d*(k*log(kappa0/(2*pi))+k*loglambda-sum(kappa0./kappa))-kappa0*aib*dot(Q(:),Q(:)));
Eqmu = 0.5*d*(k*loglambda+sum(log(kappa))-k*log(2*pi)-k);

Eplambda = k*(nu0/2*log(tau0/2)-gammaln(nu0/2))+(nu0/2-1)*sum(loglambda)-tau0*aib/2;
Eqlambda = -gammaln(nu/2)+(nu/2-1)*psi(0,nu/2)+log(tau/2)-nu/2;

xbar = bsxfun(@times,X*R,1./nk);
s = dot(sqdistance(X,xbar),R,1)./(d*nk);

r = xbar-m;
EpX = 0.5*(d*(loglambda-1./kappa-log(2*pi)-aib.*s)-aib.*dot(r,r,1))*nk';

bound = Epz-Eqz+Epw-Eqw+Epmu-Eqmu+Eplambda-Eqlambda+EpX;


function D = sqdistance(A, B)
D = (-2)*(A'*B)+bsxfun(@plus,dot(B,B,1),dot(A,A,1)');