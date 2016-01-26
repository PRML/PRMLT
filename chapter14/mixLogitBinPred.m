function t = mixLogitBinPred(model, X)
% Prediction function for mixture of logistic regression
alpha = model.alpha; % mixing coefficient
W = model.W ;  % logistic model coefficentalpha
n = size(X,2);
X = [X; ones(1,n)];
t = round(alpha*sigmoid(W'*X));

