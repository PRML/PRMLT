function model = adaboostBin(X, t)
% Adaboost for binary classification (weak learner: kmeans)
t = t+1;
k = 2;
[d,n] = size(X);
w = ones(1,n)/n;
M = 100;
Alpha = zeros(1,M);
Theta = zeros(d,k,M);
T = sparse(1:n,t,1,n,k,n);  % transform label into indicator matrix
for m = 1:M
    % weak learner
    E = spdiags(w',0,n,n)*T;
    E = E*spdiags(1./sum(E,1)',0,k,k);
    c = X*E;
    [~,y] = min(pdist2(c,X),[],1);
    Theta(:,:,m) = c;
    % adaboost
    I = y~=t;
    e = dot(w,I);
    alpha = log((1-e)/e);
    w = w.*exp(alpha*I);
    w = w/sum(w);
    Alpha(m) = alpha;
end
model.alpha = Alpha;
model.theta = Theta;