function [mu, V, llh] = kalmanFilter_(X, model)
% Square root form of Kalman filter which is numerically more stable
% Written by Mo Chen (sth4nth@gmail.com).
A = model.A; % transition matrix 
G = model.G; % transition covariance
C = model.C; % emission matrix
S = model.S;  % emision covariance
mu0 = model.mu0; % prior mean
P = model.P0;  % prior covairance