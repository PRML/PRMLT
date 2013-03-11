function [mu, V, llh] = kalmanFilter(X, model)
% DONE
% Kalman filter 
A = model.A; % transition matrix 
G = model.G; % transition covariance
C = model.C; % emission matrix
S = model.S;  % emision covariance
mu0 = model.mu0; % prior mean
P = model.P0;  % prior covairance

n = size(X,2);
q = size(mu0,1);
mu = zeros(q,n);
V = zeros(q,q,n);
llh = zeros(1,n);
I = eye(q);

PC = P*C';
R = (C*PC+S);
K = PC/R;
mu(:,1) = mu0+K*(X(:,1)-C*mu0);
V(:,:,1) = (I-K*C)*P;
llh(1) = pdfGaussLn(X(:,1),C*mu0,R);
for i = 2:n
    [mu(:,i), V(:,:,i), llh(i)] = ...
        forwardStep(X(:,i), mu(:,i-1), V(:,:,i-1), A, G, C, S, I);
end
llh = sum(llh);

function [mu, V, llh] = forwardStep(x, mu, V, A, G, C, S, I)
P = A*V*A'+G;
PC = P*C';
R = C*PC+S;
K = PC/R;
Amu = A*mu;
CAmu = C*Amu;
mu = Amu+K*(x-CAmu);
V = (I-K*C)*P;
llh = pdfGaussLn(x,CAmu,R);