function [AIC, optInd] = myaic(RSS, n_samps, n_nonzero)
%% aic
% Use the Akaike Information Criterion (BIC) 
% to select the best parameter value(s) for a regression model 
%       AIC = n*ln(RSS/n) + 2k
% n - number of samples
% RSS - residual sum of squares
% k - number of nonzero predictors
%% Inputs
% RSS - array with the residual sum of squares for each parameter value
% n_samps - number of samples
% n_nonzero - number of nonzero predictors for each parameter value
%% Outputs
% AIC - AIC for each parameter value
% optInd - index with the minimum value for BIC
%% Author: Peter DeWeirdt 
% Summer Intern Cincinnati Children's Hospital
%% Date: 6/18/2018
AIC = n_samps*arrayfun(@(x) log(x), (RSS/n_samps)) + ... 
    2*n_nonzero;
[~, optInd] = min(AIC);
end
