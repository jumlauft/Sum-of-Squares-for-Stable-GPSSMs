function [V, dVdx, dVdP] = SOS(x,P,dSOS)
% Computes the Weighted Sum of Asymmetric Quadratic Functions
% In:
%    P    Dm x Dm     Positive Symmetric matrix
%    x    E x N       Point where function is evaluated
%    dSOS 1 x 1       Degree of SOS
% OR exMat Dm x E     Combinations of exponents matrix
% Out:
%    V    N x 1       Function value
%    dVdx N x E x N       Derviative w.r.t x
%    dVdP N x Dm x Dm
% E: Dimensionality of x
% Dm: Dimensionality of monomial

% Copyright (c) by Jonas Umlauft (TUM) under BSD License 
% Last modified: Jonas Umlauft 2017-05 

[E,N] = size(x); Dm = size(P,1);

if isscalar(dSOS)
    exMat = getExpoMatrix(E,dSOS);
else
    exMat = dSOS;
end

if size(exMat,1) ~=Dm || size(P,2) ~= Dm ||size(exMat,2) ~=E
    error('wrong input dimensions');
end

if nargout == 1
    mon = getMonomial(x,exMat);
else
    [mon, dmondx]= getMonomial(x,exMat);
end


V = sum(permute(sum(permute(P,[3 1 2]).*mon',2),[1 3 2]).*mon',2);

if nargout > 1
    dVdx = zeros(E,N,N); iNN = 1:N+1:N^2;
    for e = 1:E
        dmondx_temp = permute(dmondx(:,:,e,:),[1 2 4 3]); %Dm x N x N
        dVdx(e,iNN) = 2*sum(permute(sum(permute(P,[3 1 2]).*dmondx_temp(:,iNN)',2),[1 3 2]).*mon',2);
    end
    dVdx = permute(dVdx,[2 1 3]);
end

if nargout > 2
    dVdP = zeros(N,Dm,Dm);
    for dm1=1:Dm
        for dm2=1:Dm
            dVdP(:,dm1,dm2) = dVdP(:,dm1,dm2) + (mon(dm1,:).*mon(dm2,:))';
        end
    end
end

 
% if nargout > 1
%     dVdx = zeros(E,N,N); dmondx_perm = permute(dmondx,[1 3 2 4]);
%     for dm1=1:Dm
%         for dm2=1:Dm
%             for e = 1:E
%                 dVdx(e,iNN) = dVdx(e,iNN) + 2*permute(dmondx_perm(dm2,e,iNN),[1 3 2]).*P(dm1,dm2).*mon(dm1,:);
%             end
%         end
%     end
%     dVdx = permute(dVdx,[2 1 3]);
% end

%     if nargout > 1,dVdx = zeros(N,E,N);end
%     if nargout > 1,dVdP = zeros(N,E,E);end
%     for n = 1:N
%         if nargout > 1
%             dVdx(n,:,n) = 2*permute(dmondx(:,n,:,n),[3 1 2 4])*P*mon(:,n);
%             if nargout > 2
%                 dVdP(n,:,:) = mon(:,n)*mon(:,n)';
%             end
%         end
%     end    
    
