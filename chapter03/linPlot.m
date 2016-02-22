function linPlot(model, X, t)
% Plot linear function for 1d data data
% Input:
%   model: trained model structure
%   X: 1 x n data
%   t: 1 x n response
% Written by Mo Chen (sth4nth@gmail.com).
color = [255,228,225]/255; %pink
% [x,idx] = sort(x);
x = linspace(min(X),max(X));
[y,s] = linRegPred(model,x);
figure;
hold on;
fill([x,fliplr(x)],[y+s,fliplr(y-s)],color);
plot(X,t,'o');
plot(x,y,'r-');
hold off

