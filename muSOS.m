function [EV, dEVdmu] = muSOS(mu,sigma,P,dSOS)
%MUSOS Computes expected value of a Sum of Squares (SOS)
% Computes  E[SOS(x)] = E[mon(x)' P mon(x)]  with  x ~ N(mu,sigma) and
% mon(x) are monomials of x upto degree dSOS. 
% only supports E = 2 and dSOS <= 2
% In:
%    mu             E x 1      Mean vector of input distribution
%    sigma          E x 1      Variance vector of input distribution
%    P              Dm  x Dm   Symmetric positive definite matrix
%    dSOS           1 x 1      Degree of SOS  
% OR exMat          Dm x E     Combinations of exponents matrix
% Out:
%    EV             1 x 1      Expectation of the Lyapunov function
%    dEVdmu         E x 1      Expectation of the Lyapunov function
% E:   Dimensionality of data
% Dm: Dimensionality of monomial
%
% Copyright (c) by Jonas Umlauft (TUM) under BSD License 
% Last modified: Jonas Umlauft 2017-05 

E = size(mu,1); Dm = size(P,1);

if isscalar(dSOS)
    exMat = getExpoMatrix(E,dSOS);
else
    exMat = dSOS;
end

if size(exMat,1) ~=Dm || size(P,2) ~= Dm ||size(exMat,2) ~=E ...
        || size(sigma,1) ~=E || size(sigma,2) ~=1 || size(mu,2) ~=1
    error('wrong input dimensions');
end

if E~=2 || (~(size(exMat,1)==5 && size(exMat,2)==2) && ~(size(exMat,1)==2 && size(exMat,2)==2))
    error('only supports E = 2 and dSOS<=2 ');
end


% Matrix with exponents of m(x)*m(x)^T
expo=zeros(Dm,Dm,E);
for i=1:Dm
    for j=1:Dm
        expo(i,j,:)=exMat(i,:)+exMat(j,:);
    end
end

% higher moments and derivatives of them if necessary
if dSOS==1
    moments = [1, mu(2), sigma(2)+mu(2)^2; ...
        mu(1), mu(1)*mu(2), 0;...
        sigma(1)+mu(1)^2, 0, 0];
    if nargout>1
        dmud1 = [0, 0, 0; 1, mu(2), 0; 2*mu(1), 0, 0];
        dmud2 = [0, 1, 2*mu(2); 0, mu(1), 0; 0, 0, 0];
    end
else
    moments = [1, mu(2), sigma(2)+mu(2)^2, mu(2)^3+3*mu(2)*sigma(2), ...
        3*sigma(2)^2+6*sigma(2)*mu(2)^2+mu(2)^4; ...
        mu(1), mu(1)*mu(2), mu(2)^2*mu(1)+mu(1)*sigma(2), 3*mu(2)*mu(1)*sigma(2)+mu(2)^3*mu(1), 0;...
        sigma(1)+mu(1)^2, mu(1)^2*mu(2)+mu(2)*sigma(1), ...
        sigma(1)*sigma(2)+mu(1)^2*sigma(2)+mu(2)^2*sigma(1)+mu(1)^2*mu(2)^2, 0, 0;...
        mu(1)^3+3*mu(1)*sigma(1), 3*mu(1)*mu(2)*sigma(1)+mu(1)^3*mu(2), 0, 0, 0;...
        3*sigma(1)^2+6*sigma(1)*mu(1)^2+mu(1)^4, 0, 0, 0, 0];
    if nargout>1
        dmud1 = [0, 0, 0, 0, 0; ...
            1, mu(2), mu(2)^2+sigma(2), 3*mu(2)*sigma(2)+mu(2)^3, 0; ...
            2*mu(1), 2*mu(1)*mu(2), 2*mu(1)*sigma(2)+2*mu(1)*mu(2)^2, 0, 0; ...
            3*mu(1)^2+3*sigma(1), 3*mu(2)*sigma(1)+3*mu(1)^2*mu(2), 0, 0, 0; ...
            12*sigma(1)*mu(1)+4*mu(1)^3, 0, 0, 0, 0];
        dmud2 = [0, 1, 2*mu(2), 3*mu(2)^2+3*sigma(2), 12*sigma(2)*mu(2)+4*mu(2)^3; ...
            0, mu(1), 2*mu(2)*mu(1), 3*mu(1)*sigma(2)+3*mu(2)^2*mu(1), 0; ...
            0, mu(1)^2+sigma(1), 2*mu(2)*sigma(1)+2*mu(1)^2*mu(2), 0, 0; ...
            0, 3*mu(1)*sigma(1)+mu(1)^3, 0, 0, 0; ...
            0, 0, 0, 0, 0];
    end
    
end

% transform moments into suitable matrix
moMa=zeros(Dm,Dm);
if nargout>1
    dmoMadmu=zeros(Dm,2*Dm);
end
for i=1:Dm
    for j=1:Dm
        moMa(i,j)=moments(1+expo(i,j,1),1+expo(i,j,2));
        if nargout>1
            dmoMadmu(i,j)=dmud1(1+expo(i,j,1),1+expo(i,j,2));
            dmoMadmu(i,j+Dm)=dmud2(1+expo(i,j,1),1+expo(i,j,2));
        end
    end
end

% expectation of V
EV=sum(sum(moMa.*P));

%derivative of E[V]
if nargout>1
    dEVdmu(1,1)=sum(sum(P.*dmoMadmu(:,1:Dm)));
    dEVdmu(2,1)=sum(sum(P.*dmoMadmu(:,Dm+1:end)));
end

end
