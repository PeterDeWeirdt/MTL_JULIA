include("dirty_MTL_serial.jl")

function getFitsSerial(DataMatPaths::Array{String, 1}, fit::Symbol, Smin::Float64,
    Smax::Float64, Ssteps::Int64, nB::Int64, TaskNames::Array{String,1},
    FitsOutputDir::String, FitsOutputMat::String; extrapolation::Bool = false,
    measure::Symbol = :MCC, rankCol::Int64 = 3, gsString::String = "",
    nfolds::Int64 = 2, gsTargsString::String = "")
    ntasks = length(DataMatPaths)
    nsamps = Array{Int64}(ntasks)
    ngenes = Int64
    nTFs = Int64
    taskMTLinputs = Array{Dict{String,Any},1}(ntasks)
    Xs = Array{Array{Float64,2},1}(ntasks)
    YSs = Array{Array{Float64,2},1}(ntasks)
    priors = Array{Array{Float64,2},1}(ntasks)
    println("Reading in .mat data for Fit inference")
    @showprogress for task = 1:ntasks
        inputs = matread(DataMatPaths[task])
        taskMTLinputs[task] = inputs
        Xs[task] = inputs["predictorMat"]'
        currSamps = size(Xs[task],1)
        nsamps[task] = currSamps
        priors[task] = inputs["priorWeightsMat"]'
        if task == 1
            ngenes = size(priors[1],2)
            nTFs = size(priors[1],1)
        end
        YSs[task] = inputs["responseMat"]'
    end
    geneNames = convert(Array{String,1}, vec(Array(taskMTLinputs[1]["targGenes"])))
    TFNames = convert(Array{String,1}, vec(taskMTLinputs[1]["allPredictors"]))
    if !isdir(FitsOutputDir)
        mkdir(FitsOutputDir)
    end
    println("Finding optimal lambdas")
    if fit == :ebic || fit == :bic
        optLams, lambdas, Fits, fitPlot = fit_network_ic(Xs, YSs,
            Smin = Smin, Smax = Smax, Ssteps = Ssteps, nB = nB,
            priors = priors, fit = fit)
    elseif fit == :cv
        optLams, lambda, Fits, fitPlot = fit_network_cv(Xs, YSs,
            Smin = Smin, Smax = Smax, Ssteps = Ssteps, nB = nB,
            priors = priors, fit = fit, nfolds = nfolds)
    elseif fit == :gs || throw(ArgumentError("fit not implimented yet"))
        gs = readdlm(gsString,String)
        gs = gs[2:end,1:2]
        optLams, lambdas, Fits, fitPlot = fit_network_GS(Xs, YSs, gs, geneNames, TFNames,
            Smin = Smin, Smax = Smax, Ssteps= Ssteps, nB = nB, priors = priors,
            npreds = nTFs, nsamps = nsamps, ngenes = ngenes,
            extrapolation = extrapolation, measure = measure, rankCol = rankCol,
            gsTargsString = gsTargsString)
    end
    lamS = optLams[1]
    lamB = optLams[2]
    println("Optimal lambdas - S: ", round(lamS, 3)," | B: ", round(lamB, 3))
    savefig(fitPlot, FitsOutputDir *  "Heatmap.pdf")
    matwrite(FitsOutputMat, Dict(
            "lambdas" => lambdas,
            "networkFits" => Fits,
            "optLamS" => lamS,
            "optLamB" => lamB,
            "Xs" => Xs,
            "YSs" => YSs,
            "priors" => priors,
            "TaskNames" => TaskNames,
            "geneNames" => geneNames,
            "TFNames" => TFNames
    ))
end

function getNetsSerial(FitsOutputMat::String, nboots::Int64,
    NetsOutputMat::String, NetsOutputFiles::Array{String,1})
    println("Loading optimal lambdas")
    FitMat = matread(FitsOutputMat)
    lamS = FitMat["optLamS"]
    lamB = FitMat["optLamB"]
    Xs = convert(Array{Array{Float64,2},1}, FitMat["Xs"])
    YSs = convert(Array{Array{Float64,2},1}, FitMat["YSs"])
    priors = convert(Array{Array{Float64,2},1}, FitMat["priors"])
    TaskNames = convert(Array{String, 1}, FitMat["TaskNames"])
    geneNames = convert(Array{String, 1}, FitMat["geneNames"])
    TFNames = convert(Array{String, 1}, FitMat["TFNames"])
    println("Getting network")
    sparseNets, p = getTaskNetworks(Xs, YSs, priors, lamS, lamB, TaskNames,
        nboots, geneNames, TFNames)
    for i = 1:length(sparseNets)
        writedlm(NetsOutputFiles[i], sparseNets[i])
    end
    savefig(p, replace(NetsOutputMat, ".mat", ".pdf"))
    matwrite(NetsOutputMat, Dict(
        "TaskNames" => TaskNames,
        "geneNames" => geneNames,
        "TFNames" => TFNames
    ))
end
