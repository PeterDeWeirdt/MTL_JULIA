#=
Example Pipeline
=#

#~ Global Setup ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
parallel = true
Nprocs = 3 # Will be ignored if parallel = false
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
    include("julia_fxns/getMTLSerial.jl")
end
#~ Get Fits ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if getFits || getNetworks

    Fit = :ebic # Options- :ebic, :bic, :cv, :gs
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
    Smin = 0.05
    Smax = 1. # Note: float and not int required
    Ssteps = 5 # This many steps per log10 interval
    nB = 3 # Number of lamB's for each lamS
    #Options for cv - will be ignored otherwise
    nfolds = 2
    #Options for gs - will be ignored otherwise
    extrapolation = false # whether to randomly guess for PR
    measure = :F1 #MCC, F1, or AUPR
    rankCol = 3
    gsString = "./inputs/RNAseq_inputs/priors/KC1p5_sp.tsv"
    gsTargsString = "./inputs/RNAseq_inputs/priors/goldStandardGeneLists/targGenesPR_mm9mm10.txt"
    if parallel
        getFitsParallel(DataMatPaths, Fit, Smin, Smax, Ssteps, nB, TaskNames, FitsOutputDir,
            FitsOutputMat, nfolds = nfolds, extrapolation = extrapolation, measure = measure,
            rankCol = rankCol, gsString = gsString, gsTargsString = gsTargsString)
    else
        getFitsSerial(DataMatPaths, Fit, Smin, Smax, Ssteps, nB, TaskNames, FitsOutputDir,
            FitsOutputMat, nfolds = nfolds, extrapolation = extrapolation, measure = measure,
            rankCol = rankCol, gsString = gsString, gsTargsString = gsTargsString)
    end
end
FitsTime = toc()

#~ Get Networks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if getNetworks || compareGS

    nboots = 10
    NetworkOutputDir =  MainOutputDir*join(TaskNames, "_")*"_confs_"*string(nboots)*"bootstraps/"
    NetsOutputMat = NetworkOutputDir*"Networks.mat"
    NetsOutputFiles = [NetworkOutputDir*task*".tsv" for task = TaskNames]

    if !isdir(NetworkOutputDir)
        mkdir(NetworkOutputDir)
    end
end
tic()
if getNetworks
    if parallel
        getNetsParallel(FitsOutputMat, nboots, NetsOutputMat, NetsOutputFiles)
    else
        getNetsSerial(FitsOutputMat, nboots, NetsOutputMat, NetsOutputFiles)
    end
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

    rankCol = 3
    extrapolation = false #for AUPR

    getGScomparison(NetsOutputFiles, NetsOutputMat, gsFile, gsTargsFile, rankCol,
        extrapolation, GsOutputDir)
end
compareTime = toc()
#~ Finished ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
println("Done!")
