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
% (2) Qian et al. (2013) "Glmnet for Matlab."
% http://www.stanford.edu/~hastie/glmnet_matlab/
% (3) Liu, Roeder, Wasserman (2010) "Stability Approach to Regularization 
%   Selection (StARS) for High Dimensional Graphical Models". Adv. Neural.
%   Inf. Proc.
% (4) Muller, Kurtz, Bonneau. "Generalized Stability Approach for Regularized
%   Graphical Models". 23 May 2016. arXiv.
% (5) Castro, De Veaux, Miraldi, Bonneau "Multitask learning for joint
%   inference of gene regulatory networks form several expression datasets"
%% Authors: Emily R. Miraldi, Ph.D., Divisions of Immunobiology and Biomedical
%   Informatics, Cincinnati Children's Hospital
% Peter DeWeirdt, Summer Intern, Divisions of Immunobiology and Biomedical
%   Informatics, Cincinnati Children's Hospital
%% Date: June 14, 2018 -PD

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
geneExprTFAdir = './STL_microWbulk_outputs/processedGeneExpTFA';
mkdir(geneExprTFAdir)
normGeneExprFile = './microarray_inputs/geneExpression/microarray_data_mm10.txt';
targGeneFile = './RNAseq_inputs/targRegLists/microarray_RNAseq_targetGenes.txt';
potRegFile = './RNAseq_inputs/targRegLists/microarray_RNAseq_targetRegs.txt';
tfaGeneFile = './RNAseq_inputs/targRegLists/genesForTFA.txt';
geneExprMat = fullfile(geneExprTFAdir,'geneExprGeneLists.mat');

disp('1. importGeneExpGeneLists.m')
importGeneExpGeneLists(normGeneExprFile,targGeneFile,potRegFile,...
    tfaGeneFile,geneExprMat)

%% 2. Given a prior of TF-gene interactions, estimate transcription factor 
% activities (TFAs) using prior-based TFA and TF mRNA levels
priorName = 'ATAC_Th17';
priorFile = ['./RNAseq_inputs/priors/' priorName '.tsv']; % Th17 ATAC-seq prior
edgeSS = 50;
minTargets = 3;
[xx, priorName, ext] = fileparts(priorFile);
tfaMat = fullfile(geneExprTFAdir,[priorName '_ss' num2str(edgeSS) '.mat']);

disp('2. integratePrior_estTFA.m')
integratePrior_estTFA(geneExprMat,priorFile,edgeSS,...
     minTargets, tfaMat)

%% 3. Select model parameters using a specified method

lambdaBias = .5;
tfaOpt = ''; % options are '_TFmRNA' or ''
lambdaMin = 0.02;
lambdaMax = 1;
totLogLambdaSteps = 10; % will have this many steps per log10 within bStARS lambda range
leaveOutSampleList = '';
leaveOutInf = ''; % leave out information 
alpha = 1; %elastic net parameter [0,1]. When alpha = 1 -> lasso, alpha = 0 ->ridge
method = 'ebic'; % options are ebic, aic, bic, and cv
nfolds = 10; %won't be used if method != cv
foldInf = '';
fitDir = fullfile('STL_microWbulk_outputs',strrep(['fits_' foldInf method leaveOutInf],'.','p'));
mkdir(fitDir)
netSummary = [priorName '_bias' strrep(num2str(100*lambdaBias),'.','p') tfaOpt...
    '_alpha' strrep(num2str(alpha),'.','p')];
fitOutMat = fullfile(fitDir,netSummary);
parallel = true; 
if parallel
    if isempty(gcp('nocreate'))
        mypool = parpool();
    end
end

disp('3. estimateFitTRN.m')
EstimateFitTRN(geneExprMat,tfaMat,lambdaBias,tfaOpt,...
    method,lambdaMin,lambdaMax,totLogLambdaSteps,...
    fitOutMat,leaveOutSampleList, parallel,nfolds,alpha)

%% 4. For the minimum fit score, rank TF-gene
% interactions, calculate confidences and network file for jp_gene_viz
% visualizations
priorMergedTfsFile = ['./RNAseq_inputs/priors/' priorName '_mergedTfs.txt'];
try % not all priors have merged TFs and merged TF files
    ls(priorMergedTfsFile) 
catch
    priorMergedTfsFile = '';
end
nboot = 20;
bootCut = .01; %Must show up in greater than this many bootstraps to be included in the final model
rankMethod = 'confidence'; % options are rank or confidence
selectionMethod = 'network'; % network or gene
networkDir = strrep(fitDir,'fits','networks');
mkdir(networkDir);
networkSubDir = fullfile(networkDir,[num2str(nboot) 'bootstraps_' ...
    strrep(num2str(bootCut), '.', 'p') 'cutoff_' '_rank_' rankMethod... 
    '_selection_' selectionMethod]);
mkdir(networkSubDir)
trnOutMat = fullfile(networkSubDir,netSummary);
outNetFileSparse = fullfile(networkSubDir,[netSummary '_sp.tsv']);
networkHistDir = fullfile(networkSubDir,'Histograms');
mkdir(networkHistDir)
bootsHistPdf = fullfile(networkHistDir,[netSummary '_bsHist']);

disp('4. buildTRNs_mLassoFit.m')
buildTRNs_mLassoFit(fitOutMat,tfaMat,priorMergedTfsFile,bootCut, nboot,...
    parallel, rankMethod , bootsHistPdf, trnOutMat,outNetFileSparse, selectionMethod,...
    alpha)

%% 5. Calculate precision-recall relative to KO-ChIP G.S.
gsFile = './RNAseq_inputs_overlap/priors/KC1p5_sp.tsv';
prNickName = 'KC1p5';
rankColTrn = 3;
prTargGeneFile = './RNAseq_inputs_overlap/priors/goldStandardGeneLists/targGenesPR_mm9mm10_overlap.txt';
gsRegsFile = '';
prDir = fullfile(networkSubDir,['PR_' prNickName]);
mkdir(prDir)
prMatBase = fullfile(prDir,netSummary);
prFigBase = fullfile(prDir,netSummary);

display('5. calcPRinfTRNs')
calcPRinfTRNs(outNetFileSparse,gsFile,rankColTrn,...
    prTargGeneFile,gsRegsFile,prMatBase,prFigBase)
