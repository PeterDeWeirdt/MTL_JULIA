#compare w/ STL
include("dirty_MTL.jl")
AllTaskNames = ["BulkNoBlockPrior", "scNoBlockPrior", "BulkBlockPrior", "scBlockPrior"]
# AllNetsOutputFiles = [NetsOutputFiles;["/Users/dewpz7/Documents/MTL_JULIA/outputs/Bulk_Micro_MTL/Bulk_MicroArray_confs_50bootstraps/Bulk.tsv",
#     "/Users/dewpz7/Documents/MTL_JULIA/outputs/Bulk_Micro_MTL/Bulk_MicroArray_confs_50bootstraps/MicroArray.tsv"]]
AllNetsOutputFiles = ["/Users/dewpz7/Documents/MTL_JULIA _old/julia_fxns/Allgenes_06_29_Par/Bulk_ScRNAseq_confs_20bootstraps/Bulk.tsv",
    "/Users/dewpz7/Documents/MTL_JULIA _old/julia_fxns/Allgenes_06_29_Par/Bulk_ScRNAseq_confs_20bootstraps/ScRNAseq.tsv",
    "/Users/dewpz7/Documents/MTL_JULIA _old/julia_fxns/BulkWsc_07_2_18/Bulk_ScRNAseq_confs_20bootstraps/Bulk.tsv",
    "/Users/dewpz7/Documents/MTL_JULIA _old/julia_fxns/BulkWsc_07_2_18/Bulk_ScRNAseq_confs_20bootstraps/ScRNAseq.tsv"]
NetsOutputMat = "/Users/dewpz7/Documents/MTL_JULIA _old/julia_fxns/Allgenes_06_29_Par/Bulk_ScRNAseq_confs_20bootstraps/Networks.mat"
GsOutputDir  = "./GSoutputs/"
getGScomparison(AllNetsOutputFiles, NetsOutputMat, gsFile, gsTargsFile, rankCol,
    extrapolation, GsOutputDir, inputNames = AllTaskNames)
savefig("ScNBulkBlockPrior.pdf")
