function mc = CLF(xk,m,funV,rho,minstep)
%CLF enforces stablity using control Lyapunov functions
% Calculates the next step mc for a given position xk and the Lyapunov
% function V such that the Lyapunov function is decreasing by at least
% rho*log(1+V(xk)) and the minimal stepsize is minstep while minimizing the
% difference between m (the proposed next position) and mc.
% In:
%    xk         E x 1     current position
%    m          E x 1     predicted next position
%    V          fhandle   Lyapunov function
%    rho        1 x 1     minimum negativity of decrease (default = 1e-2)
%    minstep    1 x 1     minimum step size for faster convergence (default =0.1)
% Out:
%   mc          E x 1     corrected next position
% E: Dimensionality of data
%
% Copyright (c) by Jonas Umlauft (TUM) under BSD License 
% Last modified: Jonas Umlauft 2017-05 


% Fill default value
if ~exist('rho','var'), rho = 1e-2; end
if ~exist('minstep','var'), minstep = 0.1; end

% Verfiy Sizes
E = size(xk,1);
if size(m,1)~=E || ~isscalar(rho) || ~isscalar(minstep)
    error('wrong input dimensions');
end


if checkStable(xk,m,funV,rho,minstep) <0
    mc = m;
else
    prob.options = optimoptions('fmincon','Display','off',...
        'CheckGradients',false,'GradObj','on', 'MaxIterations',100,...
        'MaxFunctionEvaluations',1e8);
    prob.solver = 'fmincon';
    prob.x0 = zeros(E,1);
    
    prob.nonlcon = @(m) checkStable(xk,m,funV,rho,minstep);
    prob.objective = @(mc) valfun(m,mc);
    
    mc = fmincon(prob);
end

end

function [e, dedmc] =valfun(m,mc)
e = (mc-m)'*(mc-m)/2;
dedmc = mc-m;
end

function [c, ceq] = checkStable(xk,m,funV,rho,minstep)
ceq = []; 
Vxk = funV(xk);
Vm = funV(m);

c=[(Vm - Vxk)+rho*log(1+Vxk); 
    minstep-(xk-m)'*(xk-m)];        % insures minimal step size
end