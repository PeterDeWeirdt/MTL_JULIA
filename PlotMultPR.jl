#= PlotMultPR.jl
Plot PR from network outputs and gold standard. Script is useful for comparing
outputs of MTL with STL.
Author: Peter DeWeirdt=#

include("./julia_fxns/dirty_MTL.jl")
AllTaskNames = ["Bulk", "Microarray"]
AllNetsOutputFiles = ["./outputs/Bulk_Micro_MTL/Bulk_MicroArray_confs_50bootstraps/Bulk.tsv",
    "./outputs/Bulk_Micro_MTL/Bulk_MicroArray_confs_50bootstraps/MicroArray.tsv"]
NetsOutputMat = "./outputs/Bulk_Micro_MTL/Bulk_MicroArray_confs_50bootstraps/Networks.mat"
gsFile = "./inputs/RNAseq_inputs/priors/KC1p5_sp.tsv"
gsTargsFile = "./inputs/RNAseq_inputs/priors/goldStandardGeneLists/targGenesPR_mm9mm10.txt"
rankCol = 3
extrapolation = true
GsOutputDir  = "./outputs/Bulk_Micro_MTL/GScomparisonWextrapolation/"
getGScomparison(AllNetsOutputFiles, NetsOutputMat, gsFile, gsTargsFile, rankCol,
    extrapolation, GsOutputDir, inputNames = AllTaskNames)
savefig("ScNBulkBlockPrior.pdf")
