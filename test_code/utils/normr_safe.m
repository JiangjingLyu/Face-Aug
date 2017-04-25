function [n,norm] = normr_safe(m)
%NORMR Normalize rows of a matrix.
%
%  Syntax
%
%    normr(M)
%
%  Description
%
%    NORMR(M) normalizes the columns of M to a length of 1.
%
%  Examples
%
%    m = [1 2; 3 4]
%    n = normr(m)
%
%  See also NORMC.

% Mark Beale, 1-31-92
% Copyright 1992-2007 The MathWorks, Inc.
% $Revision: 1.1.6.5 $  $Date: 2007/11/09 20:49:53 $

if nargin < 1,error('NNET:Arguments','Not enough input arguments.'); end

[mr,mc]=size(m);
if (mc == 1)
    norm = abs(m);
    n = sign(m);
else
    norm = sqrt(sum(m.*m,2));
    n = bsxfun(@times,m,1./norm);
end
n(isnan(n)) = 0;
