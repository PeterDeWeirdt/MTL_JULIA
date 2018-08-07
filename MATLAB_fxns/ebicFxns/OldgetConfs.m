function [geneFit, netFit] = OldgetConfs(X, Ys, Prior, Opts, method)
%% Goal: Get our best fit model for each column of Y and calculate how good our fit
% using a variety of methods, including ebic, aic, and bic
%% Inputs: 
% X -- Samples by predictors matrix
% Ys -- Response matrix, with each column corresponding to a response
% Prior -- Our prior matrix
% Opts -- Options for GLMNET
% Method -- string for ebic, aic, or bic
%% References:
% (1) Qian et al. (2013) "Glmnet for Matlab."
% http://www.stanford.edu/~hastie/glmnet_matlab/
% (2) Greenfield et al. (2013) "Robust data-driven incorporation of prior
% knowledge into the inference of dynamic regulatory netowrks"
% (3) Chen et al. (2008) "Extended bayesian information criteria for model
% selection with large model spaces"
%% Output:
% geneFit -  matrix with genes in rows and lambdas in columns
% netFit - median fit for each lambda
%% Author:
% Peter Deweirdt, Summer Intern, Divisions of Immunobiology and Biomedical
%   Informatics, Cincinnati Children's Hospital
%% Date: 6/15/2018
nResp = size(Ys, 2);
lambdaFit = zeros(nResp , size(Opts.lambda, 2));
optInds = zeros(1, nResp);
for i = 1:nResp
    currResp = Ys(:,i);
    currOpts = Opts;
    currOpts.penalty_factor = Prior(i,:); 
    lsoln = glmnet(X,currResp,'gaussian',currOpts);
    % Use output of GLMNET to check fit
    currBetas = fliplr(lsoln.beta); % flip so that the lambdas are increasing
    currA0 = fliplr(lsoln.a0');
    RSSs = sum(bsxfun(@minus,(bsxfun(@plus, X*currBetas, currA0)),...
        currResp).^2, 1); 
    n_samps = size(Ys, 1);
    n_nonzero = fliplr(lsoln.df');
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
        return
    end
    lambdaFit(i,:) = Fit; 
    optInds(i) = optInd;
end
geneFit = lambdaFit;
netFit = median(lambdaFit);

%% Plot ranges of solutions
lambdaRange = fliplr(Opts.lambda);
subplot(2,1,1)
boxplot(geneFit,'PlotStyle','compact')
title('Gene Fit')
set(gca,'XTick',1:3:size(Opts.lambda, 2), 'XTickLabel',round(lambdaRange(1:3:size(Opts.lambda, 2)), 3))
xlabel('\lambda Range')
ylabel(method)

subplot(2,1,2)
plot(lambdaRange, netFit,'o-','LineWidth',2)
title('Network Fit')
xlabel('lambda')
ylabel(method)
box on
set(gca,'YTick',min(geneFit(:):max(geneFit(:))))
set(gca,'xscale','log')
grid on
axis([lambdaRange(1) lambdaRange(end) 0 .5])
xlabel('\lambda Range')

