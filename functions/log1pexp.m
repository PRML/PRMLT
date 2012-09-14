function y = log1pexp(x)
% accurately compute y = log(1+exp(x))
% reference: Accurately Computing log(1-exp(|a|)) Martin Machler
seed = 33.3;
y = x;
idx = x<seed;
y(idx) = log1p(exp(x(idx)));
