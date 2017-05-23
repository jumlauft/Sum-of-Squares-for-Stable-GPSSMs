function comb=getExpoMatrix(E,D)
% Computes all possible combinations to choose E integers (not distinct)
% such that they add up to a number less or equal to D
% In: 
%   E    1 x 1 
%   D    1 x 1 
% Out: 
%  Comb  D+1 x ? 
% Copyright (c) by Jonas Umlauft (TUM) under BSD License 
% Last modified: Jonas Umlauft 2017-05 

comb = [];
for d = 0:D
    c = nchoosek(1:d+E-1,E-1);
    m = size(c,1);
    t = ones(m,d+E-1);
    t(repmat((1:m).',1,E-1)+(c-1)*m) = 0;
    u = [zeros(1,m);t.';zeros(1,m)];
    v = cumsum(u,1);
    comb = [comb; diff(reshape(v(u==0),E+1,m),1).'];
end
comb = comb(2:end,:);