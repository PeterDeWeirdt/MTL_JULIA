function [lamB_lamS, netFits] = getMultipleFits(Xs, YSs, Priors, inlambdaBRange, inlambdaSRange,...
    method, parallel, ntasks, Opts)
%% Goal: Get our best fit model for each cell in YSs and each column of Ys. 
% Then calculate how good our fit using a variety of methods, including ebic, aic, and bic
% Note: 'MATLAB:mir_warning_maybe_uninitialized_temporary' is turned off
% for the time being. Warned about assigning temporary variables that might
% be lost 
%% Inputs: 
% Xs -- Samples by predictors matrix for each task
% YSs -- Response matrix for each task, with each column of a single task's Y (Ys) 
%   corresponding to a response
% PriorS -- Our prior matrix for each task
% lambdaBRange -- range for block penalty
% lambdaSRange -- range for sparse penalty
% Method -- string for ebic, aic, or bic
% parralel -- boolean whether to use parralel for loop in glmnet
% Opts -- options for termination of the least_dirty algorithm
%% References:
% (1) Zhou et al. (2011) "Malsar: Multi-task learning via structural regularzation."
% (2) Greenfield et al. (2013) "Robust data-driven incorporation of prior
% knowledge into the inference of dynamic regulatory netowrks"
% (3) Chen et al. (2008) "Extended bayesian information criteria for model
% selection with large model spaces"

%% Output:
% netFits -  cell with a matrix with lambdaB in the 1st column and
% lambdaS in the 2nd and median EBIC in the 3rd for each task
%% Author:
% Peter Deweirdt, Summer Intern, Divisions of Immunobiology and Biomedical
%   Informatics, Cincinnati Children's Hospital
%% Date: 6/15/2018
nResp = size(YSs{1}, 1);
lamB_lamS = zeros(length(inlambdaBRange)*length(inlambdaSRange), 2);
netFits = cell(1,(1+ntasks)); % Mean fit for all networks + fit for each network.
lambdaBRange = fliplr(inlambdaBRange);
lambdaSRange = fliplr(inlambdaSRange);

if parallel
    parforArg = Inf;
else
    parforArg = 0;
end
parfor (i = 1:nResp, parforArg)
    % get (finite) predictor indices for each response // filter
    predInds = 1:size(Xs{1},2);
    X = cell(1,ntasks);Y = cell(1,ntasks);Prior(1,ntasks);
    for j1 = 1:ntasks
        currPrior = Priors{j1};
        currWeights = currPrior(i,:);
        % limit to predictors with finite lambda penalty for all taks (e.g., to exclude TF mRNA self-interaction loops)
        predInds = intersect(find(isfinite(currWeights)), predInds); 
    end
    for j2 = 1:ntasks
        currX = Xs{j2}
        X{j2} = zscore(currX(predInds,:)');
        CurrYs = YSs{j2}
        Y{j2} = zscore(CurrYs(i,:)');
        currPrior = Priors{j2};
        Prior{j2} = currPrior(i,predInds)'; 
    end
    
    [W, ~, ~, ~, mRSS] = Least_Dirty(X,Y, Prior, lamB, lamS, currOpts);
    % Use output of GLMNET to check fit
    currBetas = fliplr(lsoln.beta); % flip so that the lambdas are increasing
    currA0 = fliplr(lsoln.a0');
    RSSs = sum(bsxfun(@minus,(bsxfun(@plus, currPreds*currBetas, currA0)),...
        currResp).^2, 1);
    n_nonzero = fliplr(lsoln.df');
    
    n_samps = size(Ys, 2);
    if strcmp(method, 'ebic')
        gamma = 1; % Parameter for how much the 'extend portion of EBIC is weighted
        tot_preds = size(currBetas, 1);
        [Fit, optInd] = ebic(RSSs, n_samps, n_nonzero, tot_preds, gamma);
    elseif strcmp(method,  'bic')
        [Fit, optInd] = mybic(RSSs, n_samps, n_nonzero);
    elseif strcmp(method,  'aic')
        [Fit, optInd] = myaic(RSSs, n_samps, n_nonzero);
    else
        display(strcat(method, ' not yet supported'))
    end
    lambdaFit(i,:) = Fit; 
    nonzeroMat(i,:) = n_nonzero;
    optLams(i) = lambdaRange(optInd);
    optNonzero(i) = n_nonzero(optInd);
end
geneFit = lambdaFit;
netFit = median(lambdaFit);

%% Plot ranges of solutions
if size(Opts.lambda, 2) > 10
    logx = 1:floor(size(Opts.lambda, 2)/10):size(Opts.lambda, 2);
else
    logx = 1:1:size(Opts.lambda, 2);
end

set(0, 'defaultAxesFontSize', 12);
subplot(4,1,1)
boxplot(geneFit,'PlotStyle','compact')
title('Gene Fit')
set(gca,'XTick',logx, 'XTickLabel',round(lambdaRange(logx), 3))
xlabel('\lambda Range')
ylabel(method)

subplot(4,1,2)
scatter(lambdaRange, netFit, 30, mean(nonzeroMat), 'filled')
cb = colorbar;
ylabel(cb, 'num. pred.')
title('Network Fit')
xlabel('lambda')
ylabel(method)
box on
set(gca,'YTick',min(geneFit(:)):((max(geneFit(:)) - min(geneFit(:)))/4):max(geneFit(:)))
set(gca,'xscale','log')
grid on
axis([lambdaRange(1) lambdaRange(end) min(geneFit(:)) max(geneFit(:))])
xlabel('\lambda Range')

subplot(4,1,3)
binedges = round(lambdaRange(logx), 3);
histogram(optLams, binedges)
set(gca, 'XScale', 'log')
xlabel('\lambda Range')
ylabel('Count')
title('Optimal \lambda per gene')
grid on
axis tight

subplot(4,1,4)
histogram(optNonzero)
xlabel('Num. Nonzero')
ylabel('Count')
title('Optimal edges per gene')
grid on 
axis tight

