function [model, llh] = regressRvmEbCd(X, t)
% reference:
% Analysis of sparse Bayesian learning. NIPS(2002). By Faul and Tipping
% Fast marginal likelihood maximisation for sparse Bayesian models.
% AISTATS(2003). by Tipping and Faul 
[d,n] = size(X);
xbar = mean(X,2);
tbar = mean(t,2);
X = bsxfun(@minus,X,xbar);
t = bsxfun(@minus,t,tbar);

beta = 1/(0.1*var(t))^2;   % beta = 1/sigma^2
alpha = inf(d,1);
S = beta*dot(X,X,2);   
Q = beta*(X*t');
Sigma = zeros(0,0);  
mu = zeros(0,1);  
dim = zeros(0,1);
Phi = zeros(0,n);

iter = 1;
maxiter = 1000;
tol = 1e-4;
llh = -inf(1,maxiter);
indAction = zeros(d,3);   
iUse = false(d,1);
s = S; q = Q; 
for iter = 2:maxiter
    theta = q.^2-s;
    iNew = theta>0;
    
    iUpd = (iNew & iUse); % update
    iAdd = (iNew~=iUpd); % add
    iDel = (iUse~=iUpd); % del
    
    % find the next alpha that maximizes the marginal likilihood
    tllh = -inf(d,1);  % trial (temptoray) likelihood
    if any(iAdd)
        tllh(iAdd) = (Q(iAdd).^2-S(iAdd))./S(iAdd)+log(S(iAdd)./(Q(iAdd).^2));
    end
    if any(iDel)
        tllh(iDel) = Q(iDel).^2./(S(iDel)-alpha(iDel))-log1p1(-S(iDel)./alpha(iDel));
    end
    if any(iUpd)
        newAlpha = s(iUpd).^2./theta(iUpd);
        oldAlpha = alpha(iUpd);
        delta = 1./newAlpha-1./oldAlpha;
        tllh(iUpd) = Q(iUpd).^2.*delta./(S(iUpd).*delta+1)-log1p(S(iUpd).*delta);
    end
    [llh(iter),j] = max(tllh);
    if abs(llh(iter)-llh(iter-1)) < tol*llh(iter-1); break; end

    indAction(:,1) = iAdd;
    indAction(:,2) = iDel;
    indAction(:,3) = iUpd;
    
    % update parameters
    switch find(indAction(j,:))
        case 1 % Add
            alpha(j) = s(j)^2/theta(j);
            Sigma_jj = 1/(alpha(j)+S(j));
            mu_j = Sigma_jj*Q(j);
            phi_j = X(j,:);             

            v = beta*Sigma*(Phi*phi_j');   % temporary vector for common part
            off = -beta*Sigma_jj*v;
            Sigma = [Sigma+Sigma_jj*(v*v'), off; off', Sigma_jj];
            mu = [mu-mu_j*v; mu_j];
            
            e_j = phi_j-v'*Phi;
            v = beta*X*e_j';
            S = S-Sigma_jj*v.^2;
            Q = Q-mu_j*v;
            
            dim = [dim;j]; %#ok<AGROW>
        case 2  % del
            idx = (dim==j);
            alpha(j) = inf;
            Sigma_j = Sigma(:,idx);
            Sigma_jj = Sigma(idx,idx);
            mu_j = mu(idx);
            
            mu(idx) = [];
            Sigma(:,idx) = [];
            Sigma(idx,:) = [];
            
            kappa = 1/Sigma_jj;
            Sigma = Sigma-kappa*(Sigma_j*Sigma_j');                    % eq (33)
            mu = mu-kappa*mu_j*Sigma_j;                                  % eq (34)

            v = beta*X*(Phi'*Sigma_j);
            S = S+kappa*v.^2;                   % eq (35)
            Q = Q+kappa*mu_j*v; 
            
            dim(idx) = [];
        case 3 % update: 
            idx = (dim==j);
            newAlpha = s(j)^2/theta(j);
            oldAlpha = alpha(j);
            alpha(j) = newAlpha;

            Sigma_j = Sigma(:,idx);
            Sigma_jj = Sigma(idx,idx);
            mu_j = mu(idx);
            
            kappa = 1/(Sigma_jj+1/(newAlpha-oldAlpha));
            Sigma = Sigma-kappa*(Sigma_j*Sigma_j');                    % eq (33)
            mu = mu-kappa*mu_j*Sigma_j;                                  % eq (34)

            v = beta*X*(Phi'*Sigma_j);
            S = S+kappa*v.^2;                   % eq (35)
            Q = Q+kappa*mu_j*v; 
    end
    iUse = accumarray(dim,true,[d,1],@(x) x); % from Wei Li (pretty cool!)
    s = S; q = Q; % p.353 Execrcies 7.17
    alphaS = alpha(iUse)-S(iUse);
    s(iUse) = alpha(iUse).*S(iUse)./alphaS; % 7.104
    q(iUse) = alpha(iUse).*Q(iUse)./alphaS; % 7.105    

    Phi = X(iUse,:);    
    beta = (n-numel(dim)+dot(alpha(dim),diag(Sigma)))/sum((t-mu'*Phi).^2);
end
llh = llh(2:iter);
b = tbar-dot(mu,xbar(dim));

model.b = b;
model.w = mu;
model.alpha = alpha;
model.beta = beta;
