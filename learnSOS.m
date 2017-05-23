function [P, val] = learnSOS(X,Y,opt)
%%LEARNSOS Returns a pdf matrix P for Sum of Squares(SOS) Lyapunov function
% Given the Data set (X,Y) from a time discrete system x_k+1 = f(x_k)
% Y = x_k+1 , X = x_k , it finds a SOS_P (defined by P) such that the
% violation of the stability condition SOS_P(x_k+1) - SOS_P(x_k)< 0
% is minimized. If val is zero, all data points are stable for the SOS_P
%               P = arg min ramp(sum(SOS_P(x_k+1) - SOS_P(x_k)))
%                   s.t.  all eig(P) > alpha , alpha > 0
% where ramp(x) = 0 for x<0 and ramp(x) = x for x>0.
% The SOS is written  using mon = monomials(x) as SOS_P = mon'*P*mon
% It enforces all Eigenvalues of P to be larger than alpha
% In:
%   X       E  x N     Training data current step
%   Y       E  x N     Training data next step
%   opt.
%        maxP        Lower bound for Eigenvalues of P (default = 1e-8, not used)
%        minP        Lower bound for Eigenvalues of P (default = 1e-5)
%        opt         Options for optimizer fmincon
%        dSOS        degree of the SOS function (default = 2)
% Out:
%   P       Dm  x Dm   pdf matrix for SOS Lyapunov function
%   val     1  x 1     final value of optimization
% N: number of training points
% E: Dimensionality of data
% Dm: dimension of monomial

% Copyright (c) by Jonas Umlauft (TUM) under BSD License 
% Last modified: Jonas Umlauft 2017-05 

% Fill default value
if ~isfield(opt,'minP'), opt.minP = 1e-4; end
if ~isfield(opt,'maxP'), opt.maxP = 1e8; end
if ~isfield(opt,'dSOS'), opt.dSOS = 2; end
if ~isfield(opt,'opt'), warning('no optimizer options defined');end

% Verfiy Sizes
[E,N] = size(X);
if size(Y,1)~=E || size(Y,2)~=N || ~isscalar(opt.maxP) || ~isscalar(opt.minP) || ~isscalar(opt.dSOS) 
    error('wrong dimension');
end

exMat=getExpoMatrix(E,opt.dSOS);
Dm = size(exMat,1);

% Define optimization problem
prob.options = opt.opt;
prob.solver = 'fmincon';
prob.objective = @(L) fun(L,X,Y,opt.dSOS,exMat);
iL =tril(true(Dm)); L0 = randn(Dm);
prob.x0 =L0(iL(:));
prob.nonlcon = @(Lvec) con(Lvec,opt);


% Solve optimization
[Lvec, val] = fmincon(prob);

% Reconstruct Output
L = tril(ones(Dm));
L(iL(:)) = Lvec;
P=L*L';


function [f, dfdLvec] = fun(Lvec,X,Y,degree, exMat)
[D,N] = size(X);
Dm = nchoosek(degree+D,D)-1; triDm = (Dm+1)*Dm/2;

if nargout <= 1
    P = Lvec2SPD(Lvec);
        Vy = SOS(Y,P,exMat);
    Vx = SOS(X,P,exMat);
    dV =  Vy - Vx;
    iinc =  dV > 0;
    f = sum(dV(iinc));
else
    [P, dPdLvec] = Lvec2SPD(Lvec);
 
    [Vy,~,dVydP] = SOS(Y,P,exMat);
    [Vx,~,dVxdP] = SOS(X,P,exMat);
    dV =  Vy - Vx;
    iinc =  dV > 0;
    f = sum(dV(iinc));
    dfdLvec  = sum(reshape(dVydP(iinc,:,:)-dVxdP(iinc,:,:),sum(iinc),Dm^2)*...
        reshape(dPdLvec,Dm^2,triDm),1);
    
    
end



function [c, ceq,dcdLvec,dceqdLvec] = con(Lvec,opt)
% Compute number of elements and reconstruct L
triDm = numel(Lvec); Dm =-0.5+sqrt(0.25+2*triDm);
L = tril(ones(Dm)); L(L==1) = Lvec;

P = L*L';
if nargout <=2
    c = zeros(Dm,2);
    % Formulating constraint
    c(1:Dm,1) = eig(P) - opt.maxP;
    c(1:Dm,2) = -eig(P) + opt.minP;
    c = c(:);
    ceq = [];
else
    warning('no gradient implemented yet');
end




