function [X, model] = ldsRnd(d, k, n )

A = randn(k,k);
G = iwishrnd(eye(k),k);
C = randn(d,k);
S = iwishrnd(eye(d),d);
mu0 = randn(k,1);
P0 = iwishrnd(eye(k),k);

X = zeros(d,n);
z = gaussRnd(mu0,P0);              % 13.80
X(:,1) = gaussRnd(C*z,S);
for i = 2:n
    z = gaussRnd(A*z,G);           % 13.75, 13.78
    X(:,i) = gaussRnd(C*z,S);      % 13.76, 13.79
end

model.A = A; % transition matrix 
model.G = G; % transition covariance
model.C = C; % emission matrix
model.S = S;  % emision covariance
model.mu0 = mu0; % prior mean
model.P0 = P0;  % prior covairance
