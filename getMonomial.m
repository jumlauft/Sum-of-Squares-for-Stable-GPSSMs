function [mon, dmondx,dmondxdx]= getMonomial(x,exMat)
%%GETMONOMIAL computes all monomials for x and derivatives
% In:
%   x     E  x 1         input
%   ExMat Dm x E         Matrix of all combinations
% Out
%  mon      Dm x 1       all monomials
%  dmondx   Dm x D       Derivative w.r.t x
%  dmondxdx Dm x D x D   Derivative w.r.t x (twice)
%
% Copyright (c) by Jonas Umlauft (TUM) under BSD License
% Last modified: Jonas Umlauft 2017-05


[Dm,E] = size(exMat);

% Check input dimensions
if size(x,1) ~= E
    error('wrong input dimensions');
end

% Initialize variables
monall = ones(E,Dm);
deriv = zeros(E,Dm);
deriv2 = zeros(E,Dm,E);

for e1=1:E
    for dm=1:Dm
        monall(e1,dm) = x(e1)^exMat(dm,e1);
        if nargout > 1 ,deriv(e1,dm) = exMat(dm,e1)*x(e1)^(exMat(dm,e1)-1);end
        if nargout > 2
            for e2=1:E
                if (e1==e2)
                    deriv2(e1,dm,e2) = exMat(dm,e1)*(exMat(dm,e1)-1)*x(e1)^(exMat(dm,e1)-2);
                else
                    deriv2(e1,dm,e2) = exMat(dm,e1)*x(e1)^(exMat(dm,e1)-1);
                end
            end
        end
    end
end
mon = prod(monall)';

% Compute derivatives if necessary
if nargout > 1
    dmondx = zeros(Dm,E);  dmondxdx = zeros(Dm,E,E);
    for e1=1:E
        dmondx(:,e1) = deriv(e1,:).*prod(monall,1)./monall(e1,:);
        if nargout > 2
            for e2=1:E
                if (e1==e2)
                    dmondxdx(:,e1,e2)=deriv2(e1,:,e2).*prod(monall,1)./monall(e1,:);
                else
                    dmondxdx(:,e1,e2)=deriv2(e1,:,e2).*deriv2(e2,:,e1).*prod(monall,1)./monall(e1,:)./monall(e2,:);
                end
            end
        end
    end
end

end






