#compare w/ STL
include("dirty_MTL.jl")
AllTaskNames = ["BulkSTL", "MicroSTL", "BulkMTL", "MicroMTL"]
# AllNetsOutputFiles = [NetsOutputFiles;["/Users/dewpz7/Documents/MTL_JULIA/outputs/Bulk_Micro_MTL/Bulk_MicroArray_confs_50bootstraps/Bulk.tsv",
#     "/Users/dewpz7/Documents/MTL_JULIA/outputs/Bulk_Micro_MTL/Bulk_MicroArray_confs_50bootstraps/MicroArray.tsv"]]
AllNetsOutputFiles = ["/Users/dewpz7/Documents/MTL_JULIA _old/MATLAB_fxns/ExampleWorkflow/STL_bulkWmicro_outputs/networks_ebic/50bootstraps_0p01cutoff__rank_confidence_selection_network/ATAC_Th17_bias50_alpha1_sp.tsv",
    "/Users/dewpz7/Documents/MTL_JULIA _old/MATLAB_fxns/ExampleWorkflow/STL_microWbulk_outputs/networks_ebic/50bootstraps_0p01cutoff__rank_confidence_selection_network/ATAC_Th17_bias50_alpha1_sp.tsv",
    "/Users/dewpz7/Documents/MTL_JULIA/outputs/Bulk_Micro_MTL/Bulk_MicroArray_confs_50bootstraps/Bulk.tsv",
    "/Users/dewpz7/Documents/MTL_JULIA/outputs/Bulk_Micro_MTL/Bulk_MicroArray_confs_50bootstraps/MicroArray.tsv"]
getGScomparison(AllNetsOutputFiles, NetsOutputMat, gsFile, gsTargsFile, rankCol,
    extrapolation, GsOutputDir, inputNames = AllTaskNames)
savefig("MTLvSTL.pdf")
