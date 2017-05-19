function [P, val] = learnSOS(X,Y,degree,alpha)
%%LEARNSOS Returns a pdf matrix P for Sum of Squares(SOS) Lyapunov function
% Given the data set (X,Y) from a time discrete system x_k+1 = f(x_k)
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
%   degree  1  x 1     Degree of the SOS Lyapunov function
%   alpha   1  x 1     lower bound for eigenvalues of P (default = 1e-2)
% Out:
%   P       Dm  x Dm   pdf matrix for SOS Lyapunov function
%   val     1  x 1     final value of optimization
% N: number of training points
% E: Dimensionality of data
% Dm: dimension of monomial
%
% Copyright (c) by Jonas Umlauft (TUM) under BSD License 
% Last modified: Jonas Umlauft 2017-05 

% Fill default value
if ~exist('alpha','var'), alpha = 1e-2; end

% Verfiy Sizes
[E,N] = size(X);
if size(Y,1)~=E || size(Y,2)~=N || ~isscalar(degree)|| ~isscalar(alpha)
    error('wrong input dimension');
end

exMat = getExpoMatrix(E,degree);
Dm = size(exMat,1);

% Define optimization problem
prob.options = optimoptions('fmincon','Display','iter','GradObj','on',...
    'CheckGradients',false,'MaxFunctionEvaluations',1e8,'MaxIterations',1e4 );
prob.solver = 'fmincon';
prob.objective = @(L) fun2(L,X,Y,degree,exMat);
iL =tril(true(Dm)); L0 = rand(Dm);
prob.x0 =L0(iL(:));
prob.nonlcon = @(Lvec) con(Lvec,alpha);


% Solve optimization
[Lvec, val] = fmincon(prob);

% Reconstruct Output
L = tril(ones(Dm));
L(iL(:)) = Lvec;
P=L*L';



function [f, dfdLvec] = fun(Lvec,X,Y,degree, exMat)
[D,N] = size(X);
Dm = nchoosek(degree+D,D)-1; triDm = (Dm+1)*Dm/2;
itri = tril(true(Dm))==true; L = zeros(Dm);
L(itri(:)) = Lvec; Lii= find(itri);
f=0;
dfdLvec = zeros(numel(Lvec),1);

for n=1:N
    mx = getMonomial(X(:,n),exMat);
    my = getMonomial(Y(:,n),exMat);
    val  = max(my'*(L*L')*my - mx'*(L*L')*mx,0);
    if val > 0
        f = f + val/N;
        for tridm =1:triDm
            [i,j] =ind2sub([Dm Dm],Lii(tridm));
            dL = zeros(Dm); dL(i,j) = 1;
            dfdLvec(tridm) =dfdLvec(tridm) + (my'*(L*dL'+dL*L')*my - mx'*(L*dL'+dL*L')*mx)/N ;
        end
        
    end
end


function [f, dfdLvec] = fun2(Lvec,X,Y,degree, exMat)
[D,N] = size(X);
Dm = nchoosek(degree+D,D)-1; triDm = (Dm+1)*Dm/2;

f=0;
if nargout == 1
    P = Lvec2SPD(Lvec);
    for n=1:N
        dV = SOS(Y(:,n),P,exMat) - SOS(X(:,n),P,exMat);
        if dV > 0
            f = f + dV;
        end
    end
else
    dfdLvec = zeros(1,triDm);
    [P, dPdLvec] = Lvec2SPD(Lvec);
    for n=1:N
        [Vy,~,dVydP] = SOS(Y(:,n),P,exMat);
        [Vx,~,dVxdP] = SOS(X(:,n),P,exMat);
        dV =  Vy - Vx;
        if dV > 0
            f = f + dV;
            dfdLvec = dfdLvec + reshape(dVydP-dVxdP,1,Dm^2) * reshape(dPdLvec,Dm^2,triDm);
        end
    end
end



function [c, ceq,dcdLvec,dceqdLvec] = con(Lvec,alpha)
% Compute number of elements and reconstruct L
triDm = numel(Lvec); Dm =-0.5+sqrt(0.25+2*triDm);
L = tril(ones(Dm)); L(L==1) = Lvec;
itri = tril(true(Dm));
Lii= find(itri);

% Formulating constraint
[Q,lambda] = eig(L*L');
c = -diag(lambda) + alpha;
ceq = [];

if nargout > 2
    dcdLvec = zeros(Dm,triDm);
    for tridm =1:triDm
        for dm = 1:Dm
            [i,j] =ind2sub([Dm Dm],Lii(tridm));
            dL = zeros(Dm); dL(i,j) = 1;
            dcdLvec(dm,tridm) =-Q(:,dm)'*(L*dL'+dL*L')*Q(:,dm);
        end
    end
    dceqdLvec = [];
end


