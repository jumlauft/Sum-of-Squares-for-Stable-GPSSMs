function expMat = getExpoMatrix(E,D)
%GETEXPOMATRIX Computes all possible combinations to choose E integers 
% (not distinct) such that they add up to a number less or equal to D
% In: 
%   E    1 x 1 
%   D    1 x 1 
% Out: 
%  Comb  D+1 x ? 
%
% Copyright (c) by Jonas Umlauft (TUM) under BSD License 
% Last modified: Jonas Umlauft 2017-05 

if rem(E,1) ~= 0 ||  rem(D,1) ~= 0 || D<=0 || E<=0
    error('Inputs must be positive integer');
end

expMat = [];
for d = 0:D
    c = nchoosek(1:d+E-1,E-1);
    m = size(c,1);
    t = ones(m,d+E-1);
    t(repmat((1:m).',1,E-1)+(c-1)*m) = 0;
    u = [zeros(1,m);t.';zeros(1,m)];
    v = cumsum(u,1);
    expMat = [expMat; diff(reshape(v(u==0),E+1,m),1).'];
end
expMat = expMat(2:end,:);