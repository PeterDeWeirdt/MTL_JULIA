%% Th17example_setup
% Process expression data for transcription regulatory network (TRN) inference
% using multitask learning (MTL). Outputs are used in Julia pipeline. 
%% References: 
% (1) Miraldi et al. (2018) "Leveraging chromatin accessibility for 
% transcriptional regulatory network inference in T Helper 17 Cells"
% (2) Jalali, et al. (2010) "A dirty model for multi-task learning." 
% (3) Castro, De Veaux, Miraldi, Bonneau. (2018) "Multitask learning for joint
%   inference of gene regulatory networks form several expression datasets"
%% Authors: Emily R. Miraldi, Ph.D., Divisions of Immunobiology and Biomedical
%   Informatics, Cincinnati Children's Hospital
% Peter DeWeirdt, Summer Intern, Divisions of Immunobiology and Biomedical
%   Informatics, Cincinnati Children's Hospital
%%
clear all
close all
restoredefaultpath

matlabDir = './MATLAB_fxns/';

addpath(fullfile(matlabDir,'ebicFxns'))
addpath(fullfile(matlabDir,'infLassoStARS'))
addpath(fullfile(matlabDir,'glmnet'))
addpath(fullfile(matlabDir,'customMatlabFxns'))
%% 1. Import gene expression data, list of regulators, list of target genes
% into a Matlab .mat object

task_names = {'bulkWmicro','microWbulk'};
ntasks = length(task_names);
geneExprTFAmainDir = './outputs/Bulk_Micro_MTL/processedGeneExpTFA';
geneExprTFAdirs = cellfun(@(x) [geneExprTFAmainDir '/' x], task_names,...
    'UniformOutput', false);
normGeneExprFiles = {'./inputs/RNAseq_inputs/geneExpression/th17_RNAseq254_DESeq2_VSDcounts.txt',...
    './inputs/microarray_inputs/geneExpression/microarray_data_mm10.txt'};
targGeneFile = './inputs/RNAseq_inputs/targRegLists/microarray_RNAseq_targetGenes.txt';
potRegFile = './inputs/RNAseq_inputs/targRegLists/microarray_RNAseq_targetRegs.txt'; 
tfaGeneFile = './inputs/RNAseq_inputs/targRegLists/genesForTFA.txt';
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
Th17priorName = 'ATAC_Th17';
priorNames = {Th17priorName, Th17priorName};
priors = cellfun(@(x) ['./inputs/RNAseq_inputs/priors/' x '.tsv'], priorNames,...
    'UniformOutput', false); % Th17 ATAC-seq prior
edgeSS = 50;
minTargets = 3;
tfaMats = cell(1,ntasks);
for i = 1:ntasks 
    [xx, priorName, ext] = fileparts(priors{i});
    tfaMats{i} = fullfile(geneExprTFAdirs{i},[priorName '_ss' num2str(edgeSS) '.mat']);
end

disp(['2. integratePrior_estTFA.m for ' num2str(ntasks) ' tasks.'])
parfor i = 1:ntasks
    integratePrior_estTFA(geneExprMats{i},priors{i},edgeSS,...
        minTargets, tfaMats{i})
end

%% 3. Create prior weights matrices

lambdaBias = .5;
tfaOpt = ''; % options are '_TFmRNA' or ''
% Note: will attempt 'lambdaBrange' x 'lambdaSrange' parameter combinations
leaveOutSampleLists = cell(1,ntasks);
leaveOutInf = ''; % leave out information 
fitDir = fullfile(fileparts(geneExprTFAmainDir),strrep(['fitSetup' leaveOutInf],'.','p'));
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
%% Done