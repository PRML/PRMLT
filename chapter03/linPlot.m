function linPlot(model, x, t)
color = [255,228,225]/255; %pink
[y,sigma] = linPred(model,x,t);
h = 2*sigma;

figure;
hold on;
x = x(:);
y = y(:);
fill([x;flipud(x)],[y+h;flipud(y-h)],color);
plot(x,y,'r-');
hold off

