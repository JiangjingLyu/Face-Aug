function [St, Sw, Sb, m, cm, cy] = calc_covars(X, Y)
% Computes three covariance matrices (total, within-class, between-class)
% for data 'X', with class label given by 'Y'.
%
% Inputs:
%   X   - d x n matrix
%   Y   - length-n label vector
%
% Outputs:
%   St      - total covariance
%   Sw      - within-class covariance
%   Sb      - between-class covariance
%   m       - mean
%   cm      - class mean
%   cy      - label for class mean
%
% Copyright (C) 2012 by Zhen Li (zhenli3@illinois.edu).

[d, n] = size(X);
assert(length(Y) == n)

% Compute means.
m = mean(X, 2);

[cy, ~, y1] = unique(Y);
cm = zeros(d, length(cy), class(X));
cn = zeros(1, length(cy), class(X));

for i = 1:length(cy)
    idx = (y1==i);
    cm(:, i) = mean(X(:, idx), 2);
    cn(i) = nnz(idx);
end

% Compute covariances.
St = cov(X', 1);
cm1 = bsxfun(@times, bsxfun(@minus, cm, m), sqrt(cn / n));
Sb = cm1 * cm1';
Sw = St - Sb;
