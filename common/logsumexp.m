function s = logsumexp(x, dim)
% Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
%   By default dim = 1 (columns).
% Written by Michael Chen (sth4nth@gmail.com).
if nargin == 1, 
    % Determine which dimension sum will use
    dim = find(size(x)~=1,1);
    if isempty(dim), dim = 1; end
end

% subtract the largest in each dim
y = max(x,[],dim);
s = y+log(sum(exp(bsxfun(@minus,x,y)),dim));   % TODO: use log1p
i = isinf(y);
if any(i(:))
    s(i) = y(i);
end