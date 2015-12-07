function multiPlot(model, X, t)
% Plot binary classification result for 2d data
%   X: 2xn data matrix
%   t: 1xn label
W = model.W;
X = [X; ones(1,size(X,2))];
figure;
spread(X,t);
y = W'*X;
hold on;
contour(X(1,:),X(2,:),y,1);
hold off;