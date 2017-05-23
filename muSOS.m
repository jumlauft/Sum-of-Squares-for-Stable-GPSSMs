function [EV, dEVdmu] = muSOS(m,var,P,dSOS)
%MUSOS Computes expected value of a Sum of Squares (SOS)
% Computes  E[SOS(x)] = E[mon(x)' P mon(x)]  with  x ~ N(m,var) and
% mon(x) are monomials of x upto degree dSOS. var is diagonal matrix
% only supports E = 2 and dSOS <= 2
% In:
%    mu          E x N      Mean vector of input distribution
%    var         E x N      Variance vector of input distribution
%    P           Dm  x Dm   Symmetric positive definite matrix
%    dSOS        1 x 1      Degree of SOS
% OR exMat       Dm x E     Combinations of exponents matrix
% Out:
%    EV          N x 1      Expectation of the Lyapunov function
%    dEVdmu      N x E x N  derivative of EV w.r.t. mu
% E:   Dimensionality of data
% Dm: Dimensionality of monomial

% Copyright (c) by Jonas Umlauft (TUM) under BSD License
% Last modified: Jonas Umlauft 2017-05

[E,N] = size(m); Dm = size(P,1);

if isscalar(dSOS)
    exMat = getExpoMatrix(E,dSOS);
else
    exMat = dSOS;
end

if size(exMat,1) ~=Dm || size(P,2) ~= Dm ||size(exMat,2) ~=E ...
        || size(var,1) ~=E || size(var,2) ~=N
    error('wrong input dimensions');
end

if E~=2 || (~(size(exMat,1)==5 && size(exMat,2)==2) && ~(size(exMat,1)==2 && size(exMat,2)==2))
    error('only supports E = 2 and dSOS<=2 ');
end


% Matrix with exponents of m(x)*m(x)^T
expo = zeros(Dm,Dm,E);
for dm1 = 1:Dm
    for dm2 = 1:Dm
        expo(dm1,dm2,:) = exMat(dm1,:)+exMat(dm2,:);
    end
end
mp = permute(m,[1 3 2 ]); varp = permute(var,[1 3 2 ]); zs = zeros(1,1,N);
% higher moments and derivatives of them if necessary
if dSOS == 1
    moments = [ones(1,1,N), mp(2,:,:), varp(2,:,:)+mp(2,:,:).^2; ...
        mp(1,:,:), mp(1,:,:).*mp(2,:,:), zs;...
        varp(1,:,:)+mp(1,:,:).^2, zs, zs];
    if nargout > 1
        dmud1 = [zs, zs, zs; ones(1,1,N), mp(2,:,:), zs; 2*mp(1,:,:), zs, zs];
        dmud2 = [zs, ones(1,1,N), 2*mp(2,:,:); zs, mp(1,:,:), zs; zs, zs, zs];
    end
else
    moments = [ones(1,1,N), mp(2,:,:), varp(2,:,:)+mp(2,:,:).^2, mp(2,:,:).^3+3*mp(2,:,:).*varp(2,:,:), ...
        3*varp(2,:,:).^2+6*varp(2,:,:).*mp(2,:,:).^2+mp(2,:,:).^4; ...
        mp(1,:,:), mp(1,:,:).*mp(2,:,:), mp(2,:,:).^2.*mp(1,:,:)+mp(1,:,:).*varp(2,:,:), 3*mp(2,:,:).*mp(1,:,:).*varp(2,:,:)+mp(2,:,:).^3.*mp(1,:,:), zs;...
        varp(1,:,:)+mp(1,:,:).^2, mp(1,:,:).^2.*mp(2,:,:)+mp(2,:,:).*varp(1,:,:), ...
        varp(1,:,:).*varp(2,:,:)+mp(1,:,:).^2.*varp(2,:,:)+mp(2,:,:).^2.*varp(1,:,:)+mp(1,:,:).^2.*mp(2,:,:).^2, zs, zs;...
        mp(1,:,:).^3+3.*mp(1,:,:).*varp(1,:,:), 3*mp(1,:,:).*mp(2,:,:).*varp(1,:,:)+mp(1,:,:).^3.*mp(2,:,:), zs, zs, zs;...
        3*varp(1,:,:).^2+6*varp(1,:,:).*mp(1,:,:).^2+mp(1,:,:).^4, zs, zs, zs, zs];
    if nargout > 1
        dmud1 = [zs, zs, zs, zs, zs; ...
            ones(1,1,N), mp(2,:,:), mp(2,:,:).^2+varp(2,:,:), 3*mp(2,:,:).*varp(2,:,:)+mp(2,:,:).^3, zs; ...
            2*mp(1,:,:), 2*mp(1,:,:).*mp(2,:,:), 2*mp(1,:,:).*varp(2,:,:)+2*mp(1,:,:).*mp(2,:,:).^2, zs, zs; ...
            3*mp(1,:,:).^2+3*varp(1,:,:), 3*mp(2,:,:).*varp(1,:,:)+3*mp(1,:,:).^2.*mp(2,:,:), zs, zs, zs; ...
            12*varp(1,:,:).*mp(1,:,:)+4*mp(1,:,:).^3, zs, zs, zs, zs];
        dmud2 = [zs, ones(1,1,N), 2*mp(2,:,:), 3*mp(2,:,:).^2+3*varp(2,:,:), 12*varp(2,:,:).*mp(2,:,:)+4*mp(2,:,:).^3; ...
            zs, mp(1,:,:), 2*mp(2,:,:).*mp(1,:,:), 3*mp(1,:,:).*varp(2,:,:)+3*mp(2,:,:).^2.*mp(1,:,:), zs; ...
            zs, mp(1,:,:).^2+varp(1,:,:), 2*mp(2,:,:).*varp(1,:,:)+2*mp(1,:,:).^2.*mp(2,:,:), zs, zs; ...
            zs, 3*mp(1,:,:).*varp(1,:,:)+mp(1,:,:).^3, zs, zs, zs; ...
            zs, zs, zs, zs, zs];
    end
    
end

% transform moments into suitable matrix
moMa = zeros(Dm,Dm,N);
if nargout > 1, dmoMadmu = zeros(Dm,2*Dm,N); end
for dm1 = 1:Dm
    for dm2 = 1:Dm
        moMa(dm1,dm2,:) = moments(1+expo(dm1,dm2,1),1+expo(dm1,dm2,2),:);
        if nargout > 1
            dmoMadmu(dm1,dm2,:) = dmud1(1+expo(dm1,dm2,1),1+expo(dm1,dm2,2),:);
            dmoMadmu(dm1,dm2+Dm,:) = dmud2(1+expo(dm1,dm2,1),1+expo(dm1,dm2,2),:);
        end
    end
end
% expectation of V
EV = permute(sum(sum(moMa.*P,1),2),[3 1 2]);


%derivative of E[V]
if nargout > 1
    dEVdmu = zeros(E,N,N); iNN =1:N+1:N^2;
    dEVdmu(1,iNN) = sum(sum(P.*dmoMadmu(:,1:Dm,:),1),2);
    dEVdmu(2,iNN) = sum(sum(P.*dmoMadmu(:,Dm+1:end,:),1),2);
    dEVdmu = permute(dEVdmu,[2 1 3]);
end


end
