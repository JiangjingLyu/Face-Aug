function [Q, P] = calc_mdl_pq(F, Sigma)
Sb = F * F';
Sw = Sigma;
St = Sw + Sb;

inv_St = inv(St);
M = St - Sb * inv_St * Sb;
inv_M = inv(M);
Q = inv_St - inv_M;
P = inv_St * Sb * inv_M;
P = (P + P') / 2;