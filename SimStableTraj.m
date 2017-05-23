function [Xsim, CE,conv] = SimStableTraj(dxfun,dxfunstab,x0s,opt,funVar)
%SIMSTABLETRAJ Simulates trajectories starting at x0 following dynamics
% in f which get stabilized through ctrl until convergence to origin with
% ball stopX or after stopN steps
% In:
%    dxfun     fhandle      E x N -> E x N for next state
%    dxfunstab fhandle      E x N -> E x N compute stabilized next state
%    x0s       E x Ntraj    initial points
%    opt
%       stopX  1  x 1       stop criterion distance to origin (default = 1)
%       stopN  1 x 1        stop criterion number of steps (default = 1e3)
%    funVar    fhandle      E x N -> E x N for variance, triggers
%                           stochastic simulation
% Out:
%    X_sim     {Ntraj} E x ?
%    CE        Ntraj x 1     Correction effort for stabilization
%    conv      Ntraj x 1     binary variable storing which traj converged
% E: Dimensionality of data
%
% Copyright (c) by Jonas Umlauft (TUM) under BSD License
% Last modified: Jonas Umlauft 2017-05

if exist('funVar','var'), sto = true; else, sto = false; end
if ~isfield(opt,'stopX'), opt.stopX = 1; end
if ~isfield(opt,'stopN'), opt.stopN = 1e3; end

if  ~isscalar(opt.stopN) || ~isscalar(opt.stopX) 
    error('wrong input dimensions');
end

[E, Ntraj] = size(x0s);



X = zeros(E,Ntraj,opt.stopN); X(:,:,1) = x0s;
sumF = zeros(1,Ntraj);sumU = zeros(1,Ntraj);

for k=2:opt.stopN
    % Find inactive trajectories (which have already converged)
    iact = sqrt(sum(X(:,:,k-1).^2,1)) > opt.stopX;
    % If all have converged, finish simulation
    if  ~any(iact), break;  end
    
    X(:,~iact,k) = 0;
    x0 = X(:,iact,k-1);
    x1t = dxfun(x0);
    x1 = dxfunstab(x0,x1t);
    
    if sto == true
        X(:,iact,k) = x1+sqrt(funVar(x0)).*randn(size(x0));
    else
        X(:,iact,k) = x1;
    end
    
    sumF(iact) = sumF(iact) +  sqrt(sum((x1t-x0).^2,1));
    sumU(iact) = sumU(iact) + sqrt(sum((x1-x1t).^2,1));
    
end
conv=~iact;
Xsim = cell(Ntraj,1);
for n = 1:Ntraj
    Xsim{n} = permute(X(:,n,permute(any(X(:,n,:)~=0),[1 3 2])),[1 3 2]);
end
CE = sumU./sumF;