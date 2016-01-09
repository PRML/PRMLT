function [model,llh] = rvmFast2(X,t)

xbar = mean(X,2);
tbar = mean(t,2);
X = bsxfun(@minus,X,xbar);
t = bsxfun(@minus,t,tbar);

d = size(X,1);

tol = 1e-4;
maxiter = 5;
LLH = -inf(1,maxiter);

X2 = dot(X,X,2);   
Xt = X*t';
[v,j] = max(Xt.^2./X2);

beta = 1/mean(t.^2);   % Beta = 1/sigma^2
phi = X(j,:);
alpha = X2(j)/(v-1/beta);
sigma = 1/(alpha + beta*(phi*phi'));
mu = beta*sigma*phi*t';

V = beta*X*phi';
S = beta*X2-sigma*V.^2;
Q = beta*Xt-beta*sigma*Xt(j)*V;


iUse = j;
Phi = phi;
Alpha = alpha;
Sigma = sigma;
Mu = mu;
for iter = 2:maxiter
    s = S; q = Q;
    s(iUse) = alpha.*S(iUse)./(alpha-S(iUse));
    q(iUse) = alpha.*Q(iUse)./(alpha-S(iUse));
    theta = q.^2-s;
    
    iNew = find(theta>0);
    llh = -inf(d,1); 
    [iUpd,~,which] = intersect(iNew, iUse); % update    
    if ~isempty(iUpd) 
        alpha = s(iUpd).^2./theta(iUpd);
        delta = 1./alpha-1./Alpha(which);
        llh(iUpd) = Q(iUpd).^2./(S(iUpd)+1./delta)-log1p(S(iUpd).*delta);
    end
    
    iAdd = setdiff(iNew,iUpd);
    if ~isempty(iAdd)
        llh(iAdd) = (Q(iAdd).^2-S(iAdd))./S(iAdd)+log(S(iAdd)./(Q(iAdd).^2));
    end

    [LLH(iter),j] = max(llh);
    if abs(LLH(iter)-LLH(iter-1)) < tol*abs(LLH(iter)-LLH(2)); break; end
    
    if any(iUpd==j)
        act = 1;
    elseif any(iAdd==j)
        act = 2;
    else
        act = 0;
    end
    switch act
        case 1 % update
            idx = (iUse==j);
            alpha_ = s(j)^2/theta(j);

            alpha = Alpha(idx);
            Sigma_j = Sigma(:,idx);
            Sigma_jj = Sigma(idx,idx);
            mu_j = Mu(idx);
            
            delta = alpha_-alpha;
            kappa = delta/(Sigma_jj*delta+1);
            Sigma = Sigma-kappa*(Sigma_j*Sigma_j');                    % eq (33)
            Mu = Mu-kappa*mu_j*Sigma_j;                                  % eq (34)

            v = beta*X*(Phi'*Sigma_j);
            S = S+kappa*v.^2;                   % eq (35)
            Q = Q+kappa*mu_j*v; 
            
            Alpha(idx) = alpha_;
        case 2 % Add
            phi = X(j,:);
            alpha = s(j)^2/theta(j);
            sigma = 1/(alpha+S(j));
            mu = sigma*Q(j);

            v = beta*Sigma*(Phi*phi');   
            off = -beta*sigma*v;                     % ?beta
        %     off = -sigma*v;                     % ?beta
            Sigma = [Sigma+sigma*(v*v'), off; off', sigma];
            Mu = [Mu-mu*v; mu];

            e = phi-v'*Phi;
            v = beta*X*e';
            S = S-sigma*v.^2;
            Q = Q-mu*v;

            iUse = [iUse;j];
            Phi = [Phi;phi];
            Alpha = [Alpha;alpha];
        case 0
            disp('');
    end
            
end
llh = llh(2:iter);

model.index = iUse;
model.alpha = Alpha;
model.beta = beta;

