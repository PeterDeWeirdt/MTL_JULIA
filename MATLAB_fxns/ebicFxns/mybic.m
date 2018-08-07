function [BIC, optInd] = mybic(RSS, n_samps, n_nonzero)
%% bic
% Use the Bayesian Information Criterion (BIC) 
% to select the best parameter value(s) for a regression model 
%       BIC = n*ln(RSS/n) + k*ln(n) 
% n - number of samples
% RSS - residual sum of squares
% k - number of nonzero predictors
%% Inputs
% RSS - array with the residual sum of squares for each parameter value
% n_samps - number of samples
% n_nonzero - number of nonzero predictors for each parameter value
%% Outputs
% BIC - BIC for each parameter value
% optInd - index with the minimum value for BIC
%% Author: Peter DeWeirdt 
% Summer Intern Cincinnati Children's Hospital
%% Date: 6/18/2018
BIC = n_samps*arrayfun(@(x) log(x), (RSS/n_samps)) + ... 
    log(n_samps)*n_nonzero;
[~, optInd] = min(BIC);
end
