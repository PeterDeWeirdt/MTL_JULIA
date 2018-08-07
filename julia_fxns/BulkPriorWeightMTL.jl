#compare w/ STL
include("dirty_MTL.jl")
AllTaskNames = ["BulkMTL", "MicroMTL", "BulkSTL", "MicroSTL"]
AllNetsOutputFiles = [NetsOutputFiles; ["/Users/dewpz7/Documents/MTL_JULIA/MATLAB_setup/ExampleWorkflow/STL_bulkWmicro_outputs/networks_ebic/20bootstraps_0p01cutoff__rank_confidence_selection_network/ATAC_Th17_bias50_alpha1_sp.tsv",
    "/Users/dewpz7/Documents/MTL_JULIA/MATLAB_setup/ExampleWorkflow/STL_microWbulk_outputs/networks_ebic/20bootstraps_0p01cutoff__rank_confidence_selection_network/ATAC_Th17_bias50_alpha1_sp.tsv"]]
getGScomparison(AllNetsOutputFiles, NetsOutputMat, gsFile, rankCol,
    extrapolation, GsOutputDir, inputNames = AllTaskNames)
savefig("BlockPriorWeightMTL.pdf")
