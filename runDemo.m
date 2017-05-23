% Copyright (c) by Jonas Umlauft (TUM) under BSD License 
% Last modified: Jonas Umlauft 2017-05 

clc,clear, close all; rng default;
addpath(genpath('gpml'));
%% Set parameters
optLL.minP = 1e-5;   % Lower bound for EV of learned matrices
optLL.maxP = 1e8;   % Upper bound for EV of learned matrices
optLL.dSOS = 2;     % Degree of Sum of Squares
optLL.opt = optimoptions('fmincon','Display','off','GradObj','on',...
    'CheckGradients',false,'MaxFunctionEvaluations',1e8,'MaxIterations',1e3,...
    'SpecifyConstraintGradient',false);

optCLF.rho = 0.01;    % Minimum Decrease of Lyapunov function
optCLF.minstep = 1e-5;% Minimum Step size for stabilized DS
optCLF.opt = optimoptions('fmincon','Display','off','GradObj','on',...
    'CheckGradients',false, 'MaxIterations',100,'MaxFunctionEvaluations',1e8,...
    'SpecifyConstraintGradient',false);

optSim.stopN = 1e3;  % Stopping condition simulation: # of steps
optSim.stopX = 1;    % Stopping condition simulation: proximity origin



Nte = 1e2;    % Number of points in test grid
ds = 10;      % Downsampling of training data


sto = true;  % Run deterministic(sto=false) or stochastic(sto = true) case
%% Load Training Data
demos=load('Data.mat'); demos.X=demos.demos; 
Xtr = []; Ytr = []; Ndemo = length(demos.X);
for n = 1:Ndemo
    Xn = demos.X{n}(:,1:ds:end);
    Xtr = [Xtr Xn(:,1:end-1)]; 
    Ytr = [Ytr Xn(:,2:end)];
end
E = size(Xtr,1);
Xtr = [Xtr zeros(E,1)]; Ytr =[Ytr  zeros(E,1)]; dXtr = Ytr-Xtr;

%% Learn GPSSM model
disp('Training GP models...');
[mGP, varGP] = learnGPs(Xtr,Ytr);

%% Find Lyapunov functions
disp('Finding Lyapunov functions...');
[P_SOS, val_SOS] = learnSOS(Xtr,Ytr,optLL);

exMat = getExpoMatrix(E,optLL.dSOS);
if sto
    VLyap = @(x) muSOS(x,varGP(x),P_SOS,exMat);
else
    VLyap = @(x) SOS(x,P_SOS,exMat);
end
%% Evaluate Test Points
disp('Evaluating Grid Points...');
grid_min = min(Xtr,[],2);
grid_max = max(Xtr,[],2);
Nd=floor(nthroot(Nte,E));Nte = Nd^E;
Xte = ndgridj(grid_min,grid_max,Nd*ones(E,1));

% Evaluate Vector field with GP mean
mGPte = mGP(Xte);

% Evaluate Lyapunov functions
VLyapte = VLyap(Xte);

% Evaluate stabilized system
m_stab = CLF(Xte,mGPte,VLyap,optCLF);


%% Simulate Trajectories and Compare to training data
disp('Simulate Trajectories...');
x0s = cell2mat(cellfun(@(v) v(:,1), demos.X,'UniformOutput', false));

if sto
    Xsim = SimStableTraj(mGP,@(x0,x1)CLF(x0,x1,VLyap,optCLF),x0s,optSim,varGP);
else
    Xsim = SimStableTraj(mGP,@(x0,x1)CLF(x0,x1,VLyap,optCLF),x0s,optSim);
end

%% Visualize
disp('Plotting...');

Xte1 = reshape(Xte(1,:),Nd,Nd); Xte2 = reshape(Xte(2,:),Nd,Nd);
z = (max(max(VLyapte)))*exp(-10:0.8:0); streamd = 1;

figure; hold on;  axis equal;
title('Original  GPSSM');
quiver(Xtr(1,:),Xtr(2,:),dXtr(1,:),dXtr(2,:),'color','green', 'autoscale','off');
streamslice(Xte1,Xte2,reshape(mGPte(1,:),Nd,Nd)-Xte1,...
        reshape(mGPte(2,:),Nd,Nd)-Xte2,streamd);

figure; hold on;  axis equal;
title('Stabilized GPSSM')
quiver(Xtr(1,:),Xtr(2,:),dXtr(1,:),dXtr(2,:),'color','green', 'autoscale','off');
streamslice(Xte1,Xte2,reshape(m_stab(1,:),Nd,Nd)-Xte1,...
        reshape(m_stab(2,:),Nd,Nd)-Xte2,streamd);
contour(Xte1,Xte2,reshape(VLyapte,Nd,Nd),z,'k');
for n=1:Ndemo, plot(Xsim{n}(1,:),Xsim{n}(2,:),'r'); end
disp('Done');
