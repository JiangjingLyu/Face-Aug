function [Q1, Q2, P, b] = calc_mdl_pq_issym(F1, Sigma1, F2, Sigma2)
St1 = Sigma1 + F1 * F1';
St2 = Sigma2 + F2 * F2';
Sb12 = F1 * F2';

inv_St1 = inv(St1);
M = St2 - Sb12' * inv_St1 * Sb12;
inv_M = inv(M);

Q2 = inv(St2) - inv_M;
N = inv_St1 * Sb12;
P = N * inv_M;
Q1 = -(P * N');

b = sum(log(eig(St2))) - sum(log(eig(M)));  % bias term