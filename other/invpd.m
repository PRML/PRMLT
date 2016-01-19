function V = invpd(A)
% Compute invert of a positive definite matrix
%   A: a positive difinie matrix
% Written by Michael Chen (sth4nth@gmail.com).
I = eye(size(A));
[R,p] = chol(A);
if p > 0
    error('ERROR: the matrix is not positive definite.');
end
V = R\(R'\I);