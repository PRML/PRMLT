function [label, R] = mixGaussVbPred(X, model)
% Predict label and responsibility for Gaussian mixture model trained by VB.
% Input:
%   X: d x n data matrix
%   model: trained model structure outputed by the EM algirthm
% Output:
%   label: 1 x n cluster label
%   R: k x n responsibility
% Written by Mo Chen (sth4nth@gmail.com).