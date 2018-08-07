function [EBIC, optInd] = ebic(RSS, n_samps, n_nonzero, tot_preds, gamma)
%% ebic
% Use the Extended Bayesian Information Criterion (EBIC) 
% to select the best parameter value(s) for a regression model 
%       EBIC = n*ln(RSS/n) + k*ln(n) + 2(g)ln(p choose k)
% n - number of samples
% RSS - residual sum of squares
% k - number of nonzer predictors
% g - gamma (tuning param.)
% p - total number of predictors to choose from 
%% Inputs
% RSS - array with the residual sum of squares for each parameter value
% n_samps - number of samples
% n_nonzero - number of nonzero predictors for each parameter value
% tot_preds - total number of predictors to choose from
% gamma - tuning parameter
%% Outputs
% EBIC - EBIC for each parameter value
% optInd - index with the minimum value for EBIC
% ia - first index of unique nonzeros.. the lowest penalty for each nonzero
% cutoff with have the best EBIC, so no need to test them all
%% Note
% May want to supress warnings about nchoosek approximating astronomical
% numbers. 
% w = warning('query', 'last')
% id = w.identifier
% warning('off', id)
%% Author: Peter DeWeirdt 
% Summer Intern Cincinnati Children's Hospital
%% Date: 6/13/2018
EBIC = n_samps*arrayfun(@(x) log(x), (RSS/n_samps)) + ... 
    log(n_samps)*n_nonzero + ...
    2*gamma*arrayfun(@(k) getEP(tot_preds, k),...
    n_nonzero);
[~, optInd] = within1seMin(EBIC);
end

function ep = getEP(tot_preds, k)
    if k == 0 || k == tot_preds
        ep = 0; 
    else
        ep = (ApproxLnNFact(tot_preds) - (ApproxLnNFact(k) + ApproxLnNFact((tot_preds - k))));
    end
end
