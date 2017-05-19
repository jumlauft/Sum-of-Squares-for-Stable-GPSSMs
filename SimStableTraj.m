function [Xsim, CE,conv] = SimStableTraj(dxfun,dxfunstab,x0s,stopX,stopN,funVar)
%SIMSTABLETRAJ Simulates trajectories starting at x0 following dynamics 
% in f which get stabilized through ctrl until convergence to origin with
% ball stopX or after stopN steps
% In:
%    dxfun     fhandle      E x 1 -> E x 1 for next state 
%    dxfunstab fhandle      E x 1 -> E x 1 compute stabilized next state
%    x0s       E x Ntraj    initial points
%    stopX     1  x 1       stop criterion distance to origin
%    stopN     1 x 1        stop criterion number of steps
%    funVar    fhandle      E x 1 -> E x 1 for variance, triggers
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

[E, Ntraj] = size(x0s); 
if nargin < 4, stopX = 1;end
if nargin < 5, stopN = 1e4; end

Xsim = cell(Ntraj,1);
conv=zeros(Ntraj,1);
sumF = zeros(Ntraj,1);sumU = zeros(Ntraj,1);
for n = 1:Ntraj
    x0=x0s(:,n); Xsim{n}(:,1) = x0;
 
    while norm(x0) > stopX && size(Xsim{n},2)<stopN
        x1t = dxfun(x0);
        x1 = dxfunstab(x0,x1t);
        sumF(n) = sumF(n)+norm(x1t-x0);
        sumU(n) = sumU(n)+norm(x1-x1t);
        if sto==true
            x1=x1+sqrt(funVar(x0))*randn(1,1);
        end
        Xsim{n}= [Xsim{n} x1];  
        x0=x1;
    end
    conv(n) = size(Xsim{n},2)~=stopN;
end
CE = sumU./sumF;