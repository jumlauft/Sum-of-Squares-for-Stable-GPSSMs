function [V, dVdx, dVdP] = SOS(x,P,dSOS)
%SOS Computes Sum of Squares given as squares of monomials
% In:
%    P    Dm x Dm     Positive Symmetric matrix
%    x    E x 1       Point where function is evaluated
%    dSOS 1 x 1       Degree of SOS  
% OR exMat Dm x E     Combinations of exponents matrix
% Out:
%    V    1 x 1       Function value
%    dVdx E x 1       Derviative w.r.t x
% E: Dimensionality of x
% Dm: Dimensionality of monomial
%
% Copyright (c) by Jonas Umlauft (TUM) under BSD License 
% Last modified: Jonas Umlauft 2017-05 
E = size(x,1); Dm = size(P,1);

if isscalar(dSOS)
    exMat = getExpoMatrix(E,dSOS);
else
    exMat = dSOS;
end

if size(exMat,1) ~=Dm || size(P,2) ~= Dm ||size(exMat,2) ~=E
    error('wrong input dimensions');
end

if nargout ==1
    mon = getMonomial(x,exMat);
    V = mon'*P*mon;
else % If derivatives are requested
    [mon, dmondx]= getMonomial(x,exMat);
    V = mon'*P*mon;
    dVdx = 2*dmondx'*P*mon;
    dVdP = mon*mon';
end
