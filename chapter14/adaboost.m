function model = adaboost(X, t)
% Adaboost with decision stump for binary classification
% Written by Mo Chen (sth4nth@gmail.com).
n = size(X,2);
w = ones(1,n)/n;
T = 1000;
Alpha = zeros(1,T);
Theta = zeros(1,T);
E = sparse(1:n,t+1,1,n,2,n);
for it = 1:T
    % weak learner: decision stump
    m = bsxfun(@times,X,w)*E; 
    theta = mean(m,2);
    
    y = bsxfun(@gt,X,theta);
    I = bsxfun(@eq,y,t);
    j = max(I*w');
    I = I(j,:);
    
    % boosting
    e = sum(w.*I);
    alpha = log((1-e)./e);
    
    w = w.*exp(alpha*I);
    w = w/sum(w);
    
    Alpha(it) = alpha;
    Theta(it) = theta;
end
model.alpha = Alpha;
model.theta = Theta;








