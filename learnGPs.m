function [GPm,GPs2, hyp,iKn, beta] = learnGPs(X,Y,covfunc,meanfun,likfunc, sn_init, optp)
%LEARNGPS Prepares a GPSSM with SE kernel
% Given training data X,Y, it optimizes the hyperparameter for multiple
% GPs an provides inference model. 
% It links to the  GAUSSIAN PROCESS REGRESSION AND CLASSIFICATION Toolbox
% available at http://gaussianprocess.org/gpml/code.
% In:
%     X        D  x N   Input training points
%     Y        E  x N   Output training points
%     likfunc  fhandle    likelihood function from gpml
%     covfunc  fhandle    covariance function from gpml
%     meanfun  fhandle    mean function from gpml
%     sn_init  1 x 1      Initialization obeservation noise
%     optp     1 x 1      Hyp Optimization settings
% Out:
%     GPm      fhandle    E x N -> D x N   mean function
%     GPs2     fhandle    E x N -> D x N   variance function
%     hyp      struct     stuct of hyperparameters
%     iKn      N x N x E  inverse vovariance matrix of training data
%     beta     N x E      iKn*y
% E: Output dimensionality
% D: Input dimensionality
% Ntr: Number of training points
%
% Copyright (c) by Jonas Umlauft (TUM) under BSD License 
% Last modified: Jonas Umlauft 2017-05 


% Check inputs and set defaults
[E,N] = size(Y); D = size(X,1); % Needed for hyp.cov
if ~exist('sn_init','var'), sn_init = 0.1; end
if ~exist('likfunc','var'), likfunc = @likGauss; end
if ~exist('covfunc','var'), covfunc = @covSEard; end
if ~exist('meanfun','var'), meanfun = @meanZero; end
if ~exist('optp','var'), optp = -100; end

if size(X,2) ~= N ||  numel(sn_init) ~= 1
    error('wrong input dimensions');
end

% Initialize cell
GPs = cell(E,1);
for e=1:E
    % Initialize and optimize hyps
    hyp(e).lik=log(sn_init); hyp(e).cov=zeros(eval((feval(covfunc))),1);
    hyp(e) = minimize(hyp(e), @gp, optp, @infExact, meanfun, covfunc, likfunc, X', Y(e,:)');
    
    % Build inline function
    GPs{e}=@(x)gp(hyp(e), @infExact, meanfun, covfunc, likfunc, X', Y(e,:)', x');
end

GPm = @(x)  GPmfun(x,GPs);
GPs2 = @(x) GPs2fun(x,GPs,hyp,X,covfunc);

if nargout >3
    iKn = zeros(N,N,E);
    beta = zeros(N,E);
    for e=1:E
        K  = covfunc(hyp(e).cov,X');sn = exp(2*hyp(e).lik);
        iKn(:,:,e) = inv(K + sn*eye(N));
        beta(:,e) = (K + sn*eye(N))\Y(e,:)';
    end
end
end

function m = GPmfun(x,GPs)
D = length(GPs); N = size(x,2);
m = zeros(D,N);
for d=1:D
    m(d,:) = GPs{d}(x);
end
end

function [s2, ds2dx] = GPs2fun(x,GPs,hyp,Xtr,covfunc)
E = length(GPs); [D,N] = size(x);
s2 = zeros(E,N);
for d=1:E
    [~, s2(d,:)] = GPs{d}(x);
end
if nargout > 1
    ds2dx = zeros(E,N,D,N);
    if isequal(covfunc,@covLINard)
        for e = 1:E
            [k,~, dkdx] = covLINardj(hyp(e).cov,x,Xtr);
            K = covLINardj(hyp(e).cov,Xtr,Xtr);
            sn2 = exp(2*hyp(e).lik);
            l = exp(-2*hyp(e).cov);
            for n = 1:N
                ds2dx(e,n,:,n) = 2*x(:,n).*l -...
                    (2*k(n,:)/(K+sn2*eye(size(K)))*permute(dkdx(n,:,:,n),[2 3 1 4]))';
            end
        end
    else
        if isequal(covfunc,@covSEard)
            for e = 1:E
                [k,~, dkdx] = covSEardj(hyp(e).cov,x,Xtr);
                K = covSEardj(hyp(e).cov,Xtr,Xtr);
                sn2 = exp(2*hyp(e).lik);
                %l = exp(-2*hyp(e).cov);
                for n = 1:N
                    ds2dx(e,n,:,n) = -2*k(n,:)/(K+sn2*eye(size(K)))*permute(dkdx(n,:,:,n),[2 3 1 4]);
                end
            end
        else
            error('kernel not supported for derivative');
        end
    end
end


end