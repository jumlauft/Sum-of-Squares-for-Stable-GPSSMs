function  [SPD, dSPDdLvec] = Lvec2SPD(Lvec)
%LVEC2SPD Converts lower triangular matrix (as vector) to full matrix
% by reverting Cholesky decomposition SPD = L*L', provides derivatives
% In:
%    Lvec   triD x N
% Out:
%    SPD      D x D x N
%   dSPDdLvec D x D x N x triD x N
%
% Copyright (c) by Jonas Umlauft (TUM) under BSD License 
% Last modified: Jonas Umlauft 2017-05 

[triD,N] =size(Lvec);
D =-0.5+sqrt(0.25+2*triD);
if mod(D,1) ~= 0, error('Lvec has wrong domension'); end

iiL = tril(true(D));
SPD = zeros(D,D,N);
L= zeros(D,D,N);

for n=1:N
    Ltemp = zeros(D); Ltemp(iiL) = Lvec(:,n);
    L(:,:,n) = Ltemp;
    SPD(:,:,n) = L(:,:,n)*L(:,:,n)';
end

if nargout ==2
    ii = find(iiL);
    dSPDdLvec= zeros(D,D,N,triD,N);
    for n=1:N
        for i=1:triD
            [m,l] = ind2sub([D D],ii(i));
            dSPDdLvec(m,:,n,i,n) = dSPDdLvec(m,:,n,i,n) + L(:,l,n)';
            dSPDdLvec(:,m,n,i,n) = dSPDdLvec(:,m,n,i,n) + L(:,l,n);
        end
    end
end
end
