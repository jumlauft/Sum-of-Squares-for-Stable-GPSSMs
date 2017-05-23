function x = CLF(xk,m,Vfun,opt)
%CLF enforces stablity using control Lyapunov functions
% Calculates the next step mc for a given position xk and the Lyapunov
% function V such that the Lyapunov function is decreasing by at least
% rho*log(1+V(xk)) and the minimal stepsize is minstep while minimizing the
% difference between m (the proposed next position) and mc.
% In:
%    xk         E x N     current position
%    dxfun      E x N     predicted next position
%    Vfun       fhandle   Lyapunov function
%    opt.
%       rho     1 x 1     minimum negativity of decrease (default = 1e-2)
%       minstep 1 x 1     minimum step size for faster convergence (default =0.1)
%       opt               Options for optimizer fmincon
% Out:
%   mc          E x N     corrected next position
% E: Dimensionality of data
%
% Copyright (c) by Jonas Umlauft (TUM) under BSD License
% Last modified: Jonas Umlauft 2017-05


% Fill default value
if ~isfield(opt,'rho'), opt.rho = 1e-2; end
if ~isfield(opt,'minstep'), opt.minstep = 1-e1; end
if ~isfield(opt,'opt'), warning('no optimizer options defined');end


% Verfiy Sizes
[E,N] = size(xk);
if  ~isscalar(opt.rho) || ~isscalar(opt.minstep) || size(m,1)~=E || size(m,2)~=N
    error('wrong input dimensions');
end

prob.options = opt.opt;
prob.solver = 'fmincon';
prob.x0 = zeros(E,1);

% Index of unstable points
ii = find(checkStable(xk,m,Vfun,opt.rho) > 0);
x = m;

for i=1:length(ii)
    prob.nonlcon = @(m) checkStable(xk(:,ii(i)),m,Vfun,opt.rho,opt.minstep);
    
    prob.objective = @(mc) fun(m(:,ii(i)),mc);
    
    
    x(:,ii(i)) = fmincon(prob);
end

end

function [e, dedmc] = fun(m,mc)
e = (mc-m)'*(mc-m)/2;
dedmc = mc-m;
end

function [c, ceq,dcdm,dceqdm] = checkStable(x,m,Vfun,rho,minstep)
ceq = [];
if nargout > 2
    [Vm, dVmdm] = Vfun(m);
    if exist('minstep','var')
        dcdm = [dVmdm;
            2*(x-m)'];
    else
        dcdm = dVmdm;
    end
    dceqdm = [];
else
    Vm = Vfun(m);
end
Vx = Vfun(x);
if exist('minstep','var')
    c =  [Vm - Vx + rho*log(1+Vx);
        minstep-(x-m)'*(x-m)];        % insures minimal step size
else
    c =  Vm - Vx + rho*log(1+Vx);
end
end