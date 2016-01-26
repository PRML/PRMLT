function [label, energy] = kmeansPred(model, Xt)
% Prediction for kmeans clusterng
%   model: trained model structure
%   Xt: d x n testing data
% Written by Mo Chen (sth4nth@gmail.com).
[val,label] = min(pdist2(model.means, Xt));
energy = sum(val);