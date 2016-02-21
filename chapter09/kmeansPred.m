function [label, energy] = kmeansPred(model, Xt)
% Prediction for kmeans clusterng
% Input:
%   model: trained model structure
%   Xt: d x n testing data
% Output:
%   label: 1 x n cluster label
%   energy: optimization target value
% Written by Mo Chen (sth4nth@gmail.com).
[val,label] = min(sqdist(model.means, Xt));
energy = sum(val);