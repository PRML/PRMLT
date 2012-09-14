function [model, llh, Rnew] = mixRegress(X, y, k)
% mixture of linear regression model

% adding the bias term
X = [X;ones(1,size(X,2))];

[d,n] = size(X);

ridx = randperm(n);
X = X(:,ridx);
y = y(ridx);

z = ceil(k*rand(1,n));
% R = full(sparse(1:n,z,1,n,k,n)); % k x n
% initialize with random weights
R = rand(n,k);
R = bsxfun(@times, R, 1./sum(R));

W = zeros(d,k);
tol = 1e-10;
maxiter = 50000;
llh = -inf(1,maxiter);
converged = false;
t = 1;


while ~converged && t < maxiter
    t = t+1;
    % maximization
    nk = sum(R,1);
    alpha = nk/n;
    
    Xbar = bsxfun(@times, X*R,1./nk);
    ybar = y*R./nk;
    for j = 1:k
%        Xo = bsxfun(@minus,X,Xbar(:,j));
        Xo = X;
%         yo = y-ybar(j);      
        yo = y;
%        XR = bsxfun(@times,Xo,R(:,j)');
        XR = Xo * diag(R(:,j));
%        W(:,j) = (XR*Xo' + 1e-4*eye(d))\(XR*yo');
%        W(:,j) = (XR*Xo' )\(XR*yo');
        W(:,j) = (Xo * diag(R(:,j)) * y')' / (Xo * diag(R(:,j)) * Xo');
    end
%    w0 = ybar-dot(W,Xbar,1);
    w0 = 0*(ybar-dot(W,Xbar,1));
    
%    E = (bsxfun(@minus,y',w0)-X'*W).^2;
    E = (bsxfun(@minus,y',X'*W)).^2;
    beta = n/dot(R(:),E(:));

    % expectation
    logRho = (-0.5)*beta*E;
    % divide by the "beta"
    logRho = bsxfun(@plus,logRho,log(alpha./sqrt(2 * pi * beta)));
    T = logsumexp(logRho,2);
    logR = bsxfun(@minus,logRho,T);
    R = exp(logR);

    % llh(t) = sum(T)/n; % loglikelihood
    % we do not need to normalize the T.
    llh(t) = sum(T);
    % add abs to avoid fluctuation when llh(t) < llh(t+1) to stop unexpectedly
    % converged = abs(llh(t)-llh(t-1)) < tol*abs(llh(t)); 
end
llh = llh(2:t);
model.alpha = alpha; % mixing coefficient
model.beta = beta; % mixture component precision
model.W = W;  % linear model coefficent
model.w0 = w0; % linear model intersection
Rnew = zeros(size(R));
Rnew(ridx,:) = R;