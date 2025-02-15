%% example_workflow_Th17_EBIC
% Use mLASSO-EBIC to build a TRN from gene expression and prior
% information in five steps. Please refer to each function's help
% annotations for descriptions of inputs, outputs and other information.
%% Highlited Changes:
% Steps 1,2, and 5 are consistent with "example_workflow_TH17", 
% but step 3 now uses EBIC in order to select model parameters (with aic and bic as additional options)
% then in step 4 edges are ranked by confidence scores acrross bootstraps. 
%% References: 
% (1) Miraldi et al. (2018) "Leveraging chromatin accessibility for 
% transcriptional regulatory network inference in T Helper 17 Cells"
% (2) Zhou et al. (2011) "Malsar: Multi-task learning via structural regularzation."
% (3) Jalali, et al. (2010) "A dirty model for multi-task learning." 
% (4) Castro, De Veaux, Miraldi, Bonneau. (2018) "Multitask learning for joint
%   inference of gene regulatory networks form several expression datasets"
%% Authors: Emily R. Miraldi, Ph.D., Divisions of Immunobiology and Biomedical
%   Informatics, Cincinnati Children's Hospital
% Peter DeWeirdt, Summer Intern, Divisions of Immunobiology and Biomedical
%   Informatics, Cincinnati Children's Hospital
%% Date: July 30, 2018 -PD

clear all
close all
restoredefaultpath

matlabDir = '..';

addpath(fullfile(matlabDir,'ebicFxns'))
addpath(fullfile(matlabDir,'infLassoStARS'))
addpath(fullfile(matlabDir,'glmnet'))
addpath(fullfile(matlabDir,'customMatlabFxns'))
%% 1. Import gene expression data, list of regulators, list of target genes
% into a Matlab .mat object

task_names = {'bulkWsc','microWbulk'};
ntasks = length(task_names);
geneExprTFAmainDir = './outputs/processedGeneExpTFA';
geneExprTFAdirs = cellfun(@(x) [geneExprTFAmainDir '/' x], task_names,...
    'UniformOutput', false);
normGeneExprFiles = {'./RNAseq_inputs/geneExpression/th17_RNAseq254_DESeq2_VSDcounts.txt',...
    './microarray_inputs/geneExpression/microarray_data_mm10.txt'};
targGeneFile = './RNAseq_inputs/targRegLists/microarray_RNAseq_targetGenes.txt';
potRegFile = './RNAseq_inputs/targRegLists/microarray_RNAseq_targetRegs.txt'; %should be consistent in both folders
tfaGeneFile = './RNAseq_inputs/targRegLists/genesForTFA.txt';
geneExprMats = cellfun(@(x) fullfile(x,'geneExprGeneLists.mat'),...
    geneExprTFAdirs, 'UniformOutput', false);
parallel = true;
if parallel
    if isempty(gcp('nocreate'))
        mypool = parpool();
    end
end
disp(['1. importGeneExpGeneLists.m for ' num2str(ntasks) ' tasks.'])
parfor i = 1:ntasks
    mkdir(geneExprTFAdirs{i})
    importGeneExpGeneLists(normGeneExprFiles{i},targGeneFile,potRegFile,...
        tfaGeneFile,geneExprMats{i})
end

%% 2. Given a prior of TF-gene interactions, estimate transcription factor 
% activities (TFAs) using prior-based TFA and TF mRNA levels
priorName = 'ATAC_Th17';
priorFile = ['./RNAseq_inputs/priors/' priorName '.tsv']; % Th17 ATAC-seq prior
edgeSS = 50;
minTargets = 3;
[xx, priorName, ext] = fileparts(priorFile);
tfaMats = cellfun(@(x) fullfile(x,[priorName '_ss' num2str(edgeSS) '.mat']),...
    geneExprTFAdirs, 'UniformOutput', false);

disp(['2. integratePrior_estTFA.m for ' num2str(ntasks) ' tasks.'])
parfor i = 1:ntasks
    integratePrior_estTFA(geneExprMats{i},priorFile,edgeSS,...
        minTargets, tfaMats{i})
end

%% 3. Select model parameters using ebic

lambdaBias = .5;
tfaOpt = ''; % options are '_TFmRNA' or ''
% Note: will attempt 'lambdaBrange' x 'lambdaSrange' parameter combinations
leaveOutSampleLists = cell(1,ntasks);
leaveOutInf = ''; % leave out information 
fitDir = fullfile('./outputs',strrep(['fitSetup' leaveOutInf],'.','p'));
mkdir(fitDir)
netSummaries = cellfun(@(x) [x '_' priorName '_bias' strrep(num2str(100*lambdaBias),'.','p') tfaOpt],...
    task_names, 'UniformOutput', false);
fitOutMats = cellfun(@(x) fullfile(fitDir,x), netSummaries, 'UniformOutput', false);
fitOutMatsFile = cellfun(@(x) fullfile(fileparts(mfilename('fullpath')),fitDir,[x '.mat']), netSummaries, 'UniformOutput', false);
disp(fitOutMatsFile)
setupDir = '../../setup/';
mkdir(setupDir)
writetable(cell2table(fitOutMatsFile), fullfile(setupDir,'setup.txt'),'WriteVariableNames',false)
disp('3. FitSetup.m')
parfor i = 1:ntasks
    FitSetup(geneExprMats{i},tfaMats{i},lambdaBias,tfaOpt,...
        fitOutMats{i},leaveOutSampleLists{i})
end
%%
% %% 5. Calculate precision-recall relative to KO-ChIP G.S.
% gsFile = './RNAseq_inputs/priors/KC1p5_sp.tsv';
% outNetFileSparse = {'/Users/dewpz7/Documents/MTL_JULIA/ALLgenes_06_19/Big_NET_Bulk.tsv', 
%     '/Users/dewpz7/Documents/MTL_JULIA/ALLgenes_06_19/Big_NET_MicroArray.tsv'};
% networkDir = strrep(fitDir,'fitSetup','networks');
% prNickName = 'KC1p5';
% fit = 'AUPR';
% prDir = fullfile(networkDir,['PR_' prNickName fit]);
% mkdir(prDir)
% rankColTrn = 3;
% prTargGeneFile = '';
% gsRegsFile =  '';
% 
% for i = 1:ntasks
%     clf
%     prMatBase = fullfile(prDir,netSummaries{i});
%     prFigBase = fullfile(prDir,netSummaries{i});
%     display('5. calcPRinfTRNs')
%     calcPRinfTRNs(outNetFileSparse{i},gsFile,rankColTrn, ...
%         prTargGeneFile,gsRegsFile, targGeneFile, potRegFile, prMatBase,prFigBase)
% end
