function U = geig(C, S)
% Solve generalized eigen problem CU=aSU. U simultaneously diagonalize C and S.
% U'SU = I 
% This is concept verify code, not mean to be used.
[Q,A] = eig(S);

A = sqrt(A);
R = A\Q;

[V,~] = eig(R*C*R');

U = (Q'/A)*V;

