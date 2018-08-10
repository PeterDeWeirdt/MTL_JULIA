#= Th17example_inference.jl
Example Pipeline for Transcriptional Regulatory Network Inference Using
Multitask Learning

Author: Peter DeWeirdt, Summer Intern, Divisions of Immunobiology and Biomedical
    Informatics, Cincinnati Children's Hospital

References:
    Miraldi et al. (2018) "Leveraging chromatin accessibility for
            transcriptional regulatory network inference in T Helper 17 Cells"
    Castro, De Veaux, Miraldi, Bonneau. (2018) "Multitask learning for joint
            inference of gene regulatory networks form several expression datasets"
    Jalali, et al. (2010) "A dirty model for multi-task learning.
=#

#~ Global Setup ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
println("Inferring Networks")
parallel = true
Nprocs = 6 # Number of processors Will be ignored if parallel = false
getFits = true
getNetworks = true
compareGS = true
MainOutputDir = "./outputs/Bulk_Micro_MTL/"
TaskNames = ["Bulk", "MicroArray"]
if !isdir(MainOutputDir)
    mkdir(MainOutputDir)
end
if nworkers() != 1
    println("Removing ", nworkers(), " workers")
    rmprocs(workers()[1:end])
end
include("julia_fxns/dirty_MTL.jl")
if parallel
    addprocs(Nprocs)
    println(nworkers(), " workers set up")
    @everywhere include("julia_fxns/dirty_MTL.jl")
    include("julia_fxns/getMTLParallel.jl")
else
    include("julia_fxns/getMTLParallel.jl")
end

#~ Get Fits ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if getFits || getNetworks
    Fit = :ebic # Options- :ebic, :bic, :cv
    tolerance = 1e-2 # Dirty multitask lasso will stop iters when Δβ_j < tol ∀ j
    useBlockPrior = true
    FitsOutputDir =  MainOutputDir*join(TaskNames, "_")*"_lambdas_"string(Fit)*"/"
    FitsOutputMat = FitsOutputDir*"Fits.mat"
    if !isdir(FitsOutputDir)
        mkdir(FitsOutputDir)
    end
end
tic()
if getFits
    DataMatPaths = reshape(convert(Array{String,2},
        readdlm("./setup/setup.txt", ',')),2)
    Smin = 0.02 #Note: fits are slow for values less than 0.02
    Smax = 1. # Note: float required
    Ssteps = 10 # This many steps per log10 interval for lambdaS
    nB = 3 # Number of lamB's for each lamS
    getFitsParallel(DataMatPaths, Fit, Smin, Smax, Ssteps, nB, TaskNames, FitsOutputDir,
        FitsOutputMat, tolerance = tolerance, useBlockPrior = useBlockPrior)
end
FitsTime = toc()

#~ Get Networks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if getNetworks || compareGS
    nboots = 50
    NetworkOutputDir =  MainOutputDir*join(TaskNames, "_")*"_confs_"*string(nboots)*"bootstraps/"
    NetsOutputMat = NetworkOutputDir*"Networks.mat"
    NetsOutputFiles = [NetworkOutputDir*task*".tsv" for task = TaskNames]
    if !isdir(NetworkOutputDir)
        mkdir(NetworkOutputDir)
    end
end
tic()
if getNetworks
    getNetsParallel(FitsOutputMat, nboots, NetsOutputMat, NetsOutputFiles,
    tolerance = tolerance, useBlockPrior = useBlockPrior)
end
NetsTime = toc()
if parallel
    rmprocs(workers()[1:end])
end
#~ Comparare With Gold Standard ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tic()
if compareGS
    gsFile = "./inputs/RNAseq_inputs/priors/KC1p5_sp.tsv"
    gsTargsFile = "./inputs/RNAseq_inputs/priors/goldStandardGeneLists/targGenesPR_mm9mm10.txt"
    GsOutputDir =  MainOutputDir*join(TaskNames, "_")*"_"* split(gsFile, ['/','.'])[end-1] *
        "_Comparison/"
    if !isdir(GsOutputDir)
        mkdir(GsOutputDir)
    end
    rankCol = 3 # What column to rank edge predictions by
    extrapolation = false # For precision-recall, whether to extrapolate until 100% recall
    getGScomparison(NetsOutputFiles, NetsOutputMat, gsFile, gsTargsFile, rankCol,
        extrapolation, GsOutputDir)
end
compareTime = toc()
#~ Finished ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
println("Done!")
