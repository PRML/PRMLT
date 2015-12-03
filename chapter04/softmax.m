function s = softmax(x, dim)
% Softmax function
%   By default dim = 1 (columns).
% Written by Mo Chen (sth4nth@gmail.com).
if nargin == 1, 
    % Determine which dimension sum will use
    dim = find(size(x)~=1,1);
    if isempty(dim), dim = 1; end
end
s = exp(bsxfun(@minus,x,logsumexp(x,dim)));
