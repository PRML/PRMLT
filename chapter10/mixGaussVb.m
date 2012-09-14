function [label, model, L] = mixGaussVb(X, init, prior)
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
    prior.v = d+1;
    prior.M = eye(d);   % M = inv(W)
end
tol = 1e-20;
maxiter = 2000;
L = -inf(1,maxiter);
converged = false;
t = 1;

model.R = initialization(X,init);
while  ~converged && t < maxiter
    t = t+1;
    model = vmax(X, model, prior);
    model = vexp(X, model);
    L(t) = vbound(X,model,prior)/n;
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
function model = vmax(X, model, prior)
alpha0 = prior.alpha;
kappa0 = prior.kappa;
m0 = prior.m;
v0 = prior.v;
M0 = prior.M;
R = model.R;

nk = sum(R,1); % 10.51
alpha = alpha0+nk; % 10.58
nxbar = X*R;
kappa = kappa0+nk; % 10.60
m = bsxfun(@times,bsxfun(@plus,kappa0*m0,nxbar),1./kappa); % 10.61
v = v0+nk; % 10.63

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

model.alpha = alpha;
model.kappa = kappa;
model.m = m;
model.v = v;
model.M = M; % Whishart: M = inv(W)
% Done
function model = vexp(X, model)
alpha = model.alpha; % Dirichlet
kappa = model.kappa;   % Gaussian
m = model.m;         % Gasusian
v = model.v;         % Whishart
M = model.M;         % Whishart: inv(W) = V'*V

n = size(X,2);
[d,k] = size(m);

logW = zeros(1,k);
EQ = zeros(n,k);
for i = 1:k
    U = chol(M(:,:,i));
    logW(i) = -2*sum(log(diag(U)));      
    Q = (U'\bsxfun(@minus,X,m(:,i)));
    EQ(:,i) = d/kappa(i)+v(i)*dot(Q,Q,1);    % 10.64
end

ElogLambda = sum(psi(0,bsxfun(@minus,v+1,(1:d)')/2),1)+d*log(2)+logW; % 10.65
Elogpi = psi(0,alpha)-psi(0,sum(alpha)); % 10.66

logRho = (bsxfun(@minus,EQ,2*Elogpi+ElogLambda-d*log(2*pi)))/(-2); % 10.46
logR = bsxfun(@minus,logRho,logsumexp(logRho,2)); % 10.49
R = exp(logR);

model.logR = logR;
model.R = R;
% Done
function L = vbound(X, model, prior)
alpha0 = prior.alpha;
kappa0 = prior.kappa;
m0 = prior.m;
v0 = prior.v;
M0 = prior.M;

alpha = model.alpha; % Dirichlet
kappa = model.kappa;   % Gaussian
m = model.m;         % Gasusian
v = model.v;         % Whishart
M = model.M;         % Whishart: inv(W) = V'*V
R = model.R;
logR = model.logR;


[d,k] = size(m);
nk = sum(R,1); % 10.51

Elogpi = psi(0,alpha)-psi(0,sum(alpha));

Epz = dot(nk,Elogpi);
Eqz = dot(R(:),logR(:));
logCalpha0 = gammaln(k*alpha0)-k*gammaln(alpha0);
Eppi = logCalpha0+(alpha0-1)*sum(Elogpi);
logCalpha = gammaln(sum(alpha))-sum(gammaln(alpha));
Eqpi = dot(alpha-1,Elogpi)+logCalpha;
L = Epz-Eqz+Eppi-Eqpi;


U0 = chol(M0);
sqrtR = sqrt(R);
xbar = bsxfun(@times,X*R,1./nk); % 10.52

logW = zeros(1,k);
trSW = zeros(1,k);
trM0W = zeros(1,k);
xbarmWxbarm = zeros(1,k);
mm0Wmm0 = zeros(1,k);
for i = 1:k
    U = chol(M(:,:,i));
    logW(i) = -2*sum(log(diag(U)));      
    
    Xs = bsxfun(@times,bsxfun(@minus,X,xbar(:,i)),sqrtR(:,i)');
    V = chol(Xs*Xs'/nk(i));
    Q = V/U;
    trSW(i) = dot(Q(:),Q(:));  % equivalent to tr(SW)=trace(S/M)
    Q = U0/U;
    trM0W(i) = dot(Q(:),Q(:));

    q = U'\(xbar(:,i)-m(:,i));
    xbarmWxbarm(i) = dot(q,q);
    q = U'\(m(:,i)-m0);
    mm0Wmm0(i) = dot(q,q);
end

ElogLambda = sum(psi(0,bsxfun(@minus,v+1,(1:d)')/2),1)+d*log(2)+logW; % 10.65
Epmu = sum(d*log(kappa0/(2*pi))+ElogLambda-d*kappa0./kappa-kappa0*(v.*mm0Wmm0))/2;
logB0 = v0*sum(log(diag(U0)))-0.5*v0*d*log(2)-logmvgamma(0.5*v0,d);
EpLambda = k*logB0+0.5*(v0-d-1)*sum(ElogLambda)-0.5*dot(v,trM0W);

Eqmu = 0.5*sum(ElogLambda+d*log(kappa/(2*pi)))-0.5*d*k;
logB =  -v.*(logW+d*log(2))/2-logmvgamma(0.5*v,d);
EqLambda = 0.5*sum((v-d-1).*ElogLambda-v*d)+sum(logB);

EpX = 0.5*dot(nk,ElogLambda-d./kappa-v.*trSW-v.*xbarmWxbarm-d*log(2*pi));

L = L+Epmu-Eqmu+EpLambda-EqLambda+EpX;