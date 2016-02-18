function [label, model, L] = mixGaussVb(X, m, prior)
% Perform variational Bayesian inference for Gaussian mixture.
%   X: d x n data matrix
%   init: k (1 x 1) or label (1 x n, 1<=label(i)<=k) or center (d x k)
% Reference: Pattern Recognition and Machine Learning by Christopher M. Bishop (P.474)
% Written by Mo Chen (sth4nth@gmail.com).
fprintf('Variational Bayesian Gaussian mixture: running ... ');
[d,n] = size(X);
if nargin < 3
    prior.alpha = 1;
    prior.kappa = 1;
    prior.m = mean(X,2);
    prior.v = d+1;
    prior.M = eye(d);   % M = inv(W)
end
prior.logW = -2*sum(log(diag(chol(prior.M))));

tol = 1e-20;
maxiter = 5000;
L = -inf(1,maxiter);
lb = -inf(1,maxiter);
model = init(X,m);
model.nk = sum(model.R,1);
for iter = 2:maxiter
    [model, LDir] = qDir(model,prior);            % q(pi) Direchlet
    [model, LGW] = qGW(X, model, prior);          % q(mu,Lambda) GaussianWishart
    [model, LMul] = qMul(X, model,prior);         % q(z) multinomial
    lb(iter) = bound(X,model,prior);
    L(iter) = LDir+LGW+LMul;                      % lower bound
    if abs(L(iter)-L(iter-1)) < tol*abs(L(iter)); break; end
end
L = L(2:iter);
lb = lb(2:iter);
label = zeros(1,n);
[~,label(:)] = max(model.R,[],2);
[~,~,label(:)] = unique(label);
fprintf('done in %d iterations.\n',iter);

% Done
function model = init(X, m)
n = size(X,2);
if isstruct(m)  % init with a model
    model = m;
elseif numel(m) == 1  % random init k
    k = m;
    label = ceil(k*rand(1,n));
    model.R = full(sparse(1:n,label,1,n,k,n));
elseif all(size(m)==[1,n])  % init with labels
    label = m;
    k = max(label);
    model.R = full(sparse(1:n,label,1,n,k,n));
else
    error('ERROR: init is not valid.');
end

% Done
function [model, L] = qDir(model, prior)
alpha0 = prior.alpha;
nk = model.nk;

k = numel(nk);
alpha = alpha0+nk; % 10.58
ElogPi = psi(0,alpha)-psi(0,sum(alpha)); % 10.66

% lower bound
logCalpha0 = gammaln(k*alpha0)-k*gammaln(alpha0);
logCalpha = gammaln(sum(alpha))-sum(gammaln(alpha));
Eppi = logCalpha0+(alpha0-1)*sum(ElogPi);
Eqpi = logCalpha+dot(alpha-1,ElogPi);
L = Eppi-Eqpi;

model.alpha = alpha;    
model.ElogPi = ElogPi;

% Done
function [model, L] = qGW(X, model, prior)
kappa0 = prior.kappa;
m0 = prior.m;
v0 = prior.v;
M0 = prior.M;
R = model.R;
nk = model.nk;

[d,n] = size(X);
k = numel(nk);

kappa = kappa0+nk; % 10.60
m = bsxfun(@plus,X*R,kappa0*m0);
m = bsxfun(@times,m,1./kappa); % 10.61
v = v0+nk; % 10.63

r = sqrt(R');
M = zeros(d,d,k);
for i = 1:k
    Xm = bsxfun(@minus,X,m(:,i));
    Xm = bsxfun(@times,Xm,r(i,:));
    m0m = m0-m(:,i);
    M(:,:,i) = M0+Xm*Xm'+kappa0*(m0m*m0m');     % equivalent to 10.62
end

% lower bound
L = 0.5*d*(sum(log(kappa0./kappa))-n*log(2*pi));
model.kappa = kappa;
model.m = m;
model.v = v;
model.M = M; % Whishart: M = inv(W)

% Done
function [model, L] = qMul(X, model,prior)
ElogPi = model.ElogPi;
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
    Q = U'\bsxfun(@minus,X,m(:,i));
    EQ(:,i) = d/kappa(i)+v(i)*dot(Q,Q,1);    % 10.64
end
ElogLambda = sum(psi(0,bsxfun(@minus,v+1,(1:d)')/2),1)+d*log(2)+logW; % 10.65
logPx = -0.5*bsxfun(@minus,EQ,ElogLambda)-0.5*d*log(2*pi); 
logRho = bsxfun(@plus,logPx,ElogPi);   % 10.46
T = logsumexp(logRho,2);
logR = bsxfun(@minus,logRho,T); % 10.49
R = exp(logR);
nk = sum(R,1); % 10.51

% lower bound
Epz = dot(nk,ElogPi);
Eqz = dot(R(:),logR(:));

v0 = prior.v;
logW0 = prior.logW;
logB0 = -0.5*v0*(logW0+d*log(2))-logMvGamma(0.5*v0,d);
logB = -0.5*v.*(logW+d*log(2))-logMvGamma(0.5*v,d);

L = Epz-Eqz+k*logB0-sum(logB);

model.logR = logR;
model.R = R;
model.nk = nk;

% Done
function L = bound(X, model, prior)
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
nk = model.nk;

[d,k] = size(m);

Elogpi = psi(0,alpha)-psi(0,sum(alpha));

Epz = dot(nk,Elogpi);
Eqz = dot(R(:),logR(:));
logCalpha0 = gammaln(k*alpha0)-k*gammaln(alpha0);
Eppi = logCalpha0+(alpha0-1)*sum(Elogpi);
logCalpha = gammaln(sum(alpha))-sum(gammaln(alpha));
Eqpi = logCalpha+dot(alpha-1,Elogpi);
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
logB0 = v0*sum(log(diag(U0)))-0.5*v0*d*log(2)-logMvGamma(0.5*v0,d);
EpLambda = k*logB0+0.5*(v0-d-1)*sum(ElogLambda)-0.5*dot(v,trM0W);

Eqmu = 0.5*sum(ElogLambda+d*log(kappa/(2*pi)))-0.5*d*k;
logB =  -v.*(logW+d*log(2))/2-logMvGamma(0.5*v,d);
EqLambda = 0.5*sum((v-d-1).*ElogLambda-v*d)+sum(logB);

EpX = 0.5*dot(nk,ElogLambda-d./kappa-v.*trSW-v.*xbarmWxbarm-d*log(2*pi));

L = L+Epmu-Eqmu+EpLambda-EqLambda+EpX;