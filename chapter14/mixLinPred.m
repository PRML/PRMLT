function [y, z, p] = mixLinPred(model, X, t)
% Prediction function for mxiture of linear regression
% input:
%   model: trained model structure
%   X: dxn data matrix
%   t:(optional) 1xn responding vector
% output:
%   y: prediction 
%   z: cluster label
%   p: probability for t
W = model.W;
alpha = model.alpha;
beta = model.beta;

X = [X;ones(1,size(X,2))]; % adding the bias term
y = W'*X;
D = bsxfun(@minus,y,t).^2;
logRho = (-0.5)*beta*D;
logRho = bsxfun(@plus,logRho,log(alpha));
T = logsumexp(logRho,1);
p = exp(T);
logR = bsxfun(@minus,logRho,T);
R = exp(logR);
z = max(R,[],1);
