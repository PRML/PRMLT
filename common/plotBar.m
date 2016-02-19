function plotBar(model, X, t)
% Plot linear function and data
% Input:
%   X: 1 x n data
%   t: 1 x n response
% Written by Mo Chen (sth4nth@gmail.com).
color = [255,228,225]/255; %pink
% [x,idx] = sort(x);
x = linspace(min(X),max(X));
[y,s] = linPred(model,x);
figure;
hold on;
fill([x,fliplr(x)],[y+s,fliplr(y-s)],color);
plot(X,t,'o');
plot(x,y,'r-');
hold off

