function binPlot(model, X, t)
% Plot binary classification result for 2d data
%   X: 2xn data matrix
%   y: 1xn label

w = model.w;
w0 = model.w0;
figure;
spread(X,t);
y = w'*X+w0;
hold on;
contour(X(1,:),X(2,:),y,1);
hold off;