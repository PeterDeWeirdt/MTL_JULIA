function [confs, rnk] = calc_confs(B, X, Y)
%% Description:
% Caclulate confidence scores for each entry in B, beta. Confidence
% scores are calculated as:
%           1 - sigma^2/sigma~i^2.
% Where sigma^2 is the variance of the residuals for the whole model, and
% sigma~i^2 is the variance of the residuals of the model without the ith
% predictor. 
%% Inputs:
% B - coefficient matrix
% X - predictor matrix
% Y - response matrix
% Note: We assume the input data is centered, so we don't need an intercept
%% Output:
% confs - the confidence score for each coefficient
% rnk - vector of ranks for each edge. 0 indicates the coefficient we are
% least confident in
%% Author:
% Peter DeWeirdt - Cincinnati Children's Summer Intern
%% Date: 6/13/2018

sigma = var((X*B) - Y);
nzero = find(B)';
confs = zeros(size(B, 1),1);
for i = nzero
    Bno_i = B;
    Bno_i(i) = 0;
    sigma_i =  var((X*Bno_i) - Y);
    confs(i) = 1 - (sigma/sigma_i);
end
[~,~,rnk] = unique(confs);
rnk = rnk - 1;
end

