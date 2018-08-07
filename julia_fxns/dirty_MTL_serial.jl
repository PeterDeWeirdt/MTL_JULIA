include("dirty_MTL.jl")

function fit_network_ic(Xs::Array{Array{Float64,2},1},
    YSs::Array{Array{Float64,2},1}; Smin::Float64 = 0.01, Smax::Float64 = 1.,
    Ssteps::Int64 = 10, nB::Int64 = 3, priors::Array{Array{Float64,2},1} = Array{Array{Float64,2},1}(0),
    fit::Symbol = :ebic, npreds::Int64 = 0, nsamps::Array{Int64,1} = Array{Int64,1}(0), ngenes::Int64 = 0)
    """Calculate ebic or bic for each gene in a network. Plot grid of median fits.
    Return lambdaS and lambdaB within 1 standard error of the minimum fit."""
    ntasks = length(Xs)
    if npreds==0;npreds = size(Xs[1], 2);end
    if length(nsamps)==0
        nsamps = Array{Int64, 1}(ntasks)
        for task = 1:ntasks
            nsamps[task] = size(Xs[task], 1)
        end
    end
    if ngenes==0;ngenes = size(YSs[1],2);end
    lamSlog10step = 1/Ssteps
    logLamSRange = log10(Smin):lamSlog10step:log10(Smax)
    lamSs = 10.^logLamSRange
    geneFits = Array{Float64,2}(length(lamSs)*nB,ngenes)
    lambdas = Array{Float64,2}(length(lamSs)*nB,2)
    Xs,YSs = preprocess_data(Xs, YSs)
    ~,Ds = covariance_update_terms(Xs, Array{Array{Float64,1},1}(0), calcCs = false, calcDs = true)
    println("Estimating fits for " * string(ngenes) * " genes")
    @showprogress for genei = 1:ngenes
        Ys = Array{Array{Float64,1},1}(ntasks)
        P = Array{Float64,2}(npreds, ntasks)
        for k = 1:ntasks
            Ys[k] = YSs[k][:,genei]
            if length(priors) != 0
                P[:,k] = priors[k][:,genei]
            else
                P[:,k] = ones(npreds)
            end
        end
        Fits, lambdas = fit_gene_ic(Xs, Ys, Ds,
        ntasks, npreds, nsamps, lamSs, nB = nB, prior = P, fit = fit)
        geneFits[:,genei] = Fits
    end
    networkFits = median(geneFits,2)
    chosenLams, fitsPlot = get_fit_outputs(geneFits, networkFits,
        lambdas, fit)
    return chosenLams, lambdas, networkFits, fitsPlot
end

function fit_network_cv(Xs::Array{Array{Float64,2},1},
    YSs::Array{Array{Float64,2},1}; Smin::Float64 = 0.01, Smax::Float64 = 1.,
    Ssteps::Int64 = 10, nB::Int64 = 3, priors::Array{Array{Float64,2},1} = Array{Array{Float64,2},1}(0),
    fit::Symbol = :ebic, npreds::Int64 = 0, nsamps::Array{Int64,1} = Array{Int64,1}(0), ngenes::Int64 = 0, nfolds::Int64 = 0)
    """Calculate cross validation fit for each gene in a network. Plot grid of median fits.
    Return lambdaS and lambdaB within 1 standard error of the minimum fit."""
    ntasks = length(Xs)
    if npreds==0;npreds = size(Xs[1], 2);end
    if length(nsamps)==0
        nsamps = Array{Int64, 1}(ntasks)
        for task = 1:ntasks
            nsamps[task] = size(Xs[task], 1)
        end
    end
    if ngenes==0;ngenes = size(YSs[1],2);end
    lamSlog10step = 1/Ssteps
    logLamSRange = log10(Smin):lamSlog10step:log10(Smax)
    lamSs = 10.^logLamSRange
    geneFits = Array{Float64,2}(length(lamSs)*nB,ngenes)
    lambdas = Array{Float64,2}(length(lamSs)*nB,2)
    folds = Array{Array{Array{Int64,1},1},1}(ntasks)
    for task = 1:ntasks
        folds[task] = kfoldperm(nsamps[task], nfolds)
    end
    foldLOXs = Array{Array{Array{Float64,2},1},1}(nfolds)
    foldInXs = Array{Array{Array{Float64,2},1},1}(nfolds)
    foldLOYSs = Array{Array{Array{Float64,2},1},1}(nfolds)
    foldInYSs = Array{Array{Array{Float64,2},1},1}(nfolds)
    foldDs = Array{Array{Array{Float64,2},1},1}(nfolds)
    for fold = 1:nfolds
        LOXs = Array{Array{Float64,2},1}(ntasks)
        LOYSs = Array{Array{Float64,2},1}(ntasks)
        InXs = Array{Array{Float64,2},1}(ntasks)
        InYSs = Array{Array{Float64,2},1}(ntasks)
        for task = 1:ntasks
            currFold = folds[task][fold]
            LOXs[task] = Xs[task][currFold,:]
            LOYSs[task] = YSs[task][currFold,:]
            keepInds = (1:nsamps[task])[filter((x) -> !(x in currFold),1:nsamps[task])]
            InXs[task] = Xs[task][keepInds,:]
            InYSs[task] = YSs[task][keepInds,:]
        end
        b4LOXs = deepcopy(LOXs)
        LOXs, LOYSs = preprocess_data(LOXs, LOYSs)
        InXs, InYSs = preprocess_data(InXs, InYSs)
        foldLOXs[fold] = LOXs
        foldInXs[fold] = InXs
        foldLOYSs[fold] = LOYSs
        foldInYSs[fold] = InYSs
        ~,foldDs[fold] = covariance_update_terms(InXs, Array{Array{Float64,1},1}(0), calcCs = false, calcDs = true)
    end
    LOXs = Array{Array{Float64,2},1}(ntasks)
    LOYSs = Array{Array{Float64,2},1}(ntasks)
    InXs = Array{Array{Float64,2},1}(ntasks)
    InYSs = Array{Array{Float64,2},1}(ntasks)
    println("Estimating fits for " * string(ngenes) * " genes")
    @showprogress for genei = 1:ngenes
        P = Array{Float64,2}(npreds, ntasks)
        for k = 1:ntasks
            if length(priors) != 0
                P[:,k] = priors[k][:,genei]
            else
                P[:,k] = ones(npreds)
            end
        end
        foldInYs = Array{Array{Array{Float64,1},1},1}(nfolds)
        foldLOYs = Array{Array{Array{Float64,1},1},1}(nfolds)
        for fold = 1:nfolds
            InYs = Array{Array{Float64,1},1}(ntasks)
            LOYs = Array{Array{Float64,1},1}(ntasks)
            for task = 1:ntasks
                InYs[task] =  foldInYSs[fold][task][:,genei]
                LOYs[task] =  foldLOYSs[fold][task][:,genei]
            end
            foldInYs[fold] = InYs
            foldLOYs[fold] = LOYs
        end
        Fits, lambdas = fit_gene_cv(foldInXs,foldInYs,foldLOXs, foldLOYs,
        foldDs, ntasks, npreds, nsamps, lamSs,nB = nB, prior = P)
        geneFits[:,genei] = Fits
    end
    networkFits = median(geneFits,2)
    chosenLams, fitsPlot = get_fit_outputs(geneFits, networkFits,
        lambdas, fit)
    return chosenLams, lambdas, networkFits, fitsPlot
end

function fit_network_GS(Xs::Array{Array{Float64,2},1},
    YSs::Array{Array{Float64,2},1}, gs::Array{String,2}, geneNames::Array{String,1},
    TFnames::Array{String,1}; Smin::Float64 = 0.01, Smax::Float64 = 1.,
    Ssteps::Int64 = 10, nB::Int64 = 4, priors::Array{Array{Float64,2},1} = Array{Array{Float64,2},1}(0),
    npreds::Int64 = 0, nsamps::Array{Int64,1} = Array{Int64,1}(0), ngenes::Int64 = 0,
    extrapolation::Bool = true, measure::Symbol = :MCC, rankCol::Int64 = 3,
    gsTargsString::String = "")
    """Calculate fit for each gene in a network. Plot grid of median fits.
    Return lambdaS and lambdaB within 1 se of the minimum fit."""
    ntasks = length(Xs)
    if npreds==0;npreds = size(Xs[1], 2);end
    if length(nsamps)==0
        nsamps = Array{Int64, 1}(ntasks)
        for task = 1:ntasks
            nsamps[task] = size(Xs[task], 1)
        end
    end
    if ngenes==0;ngenes = size(YSs[1],2);end
    lamSlog10step = 1/Ssteps
    logLamSRange = log10(Smin):lamSlog10step:log10(Smax)
    lamSs = 10.^logLamSRange
    lambdas = Array{Float64,2}(length(lamSs)*nB,2)
    Xs,YSs = preprocess_data(Xs, YSs)
    ~,Ds = covariance_update_terms(Xs, Array{Array{Float64,1},1}(0), calcCs = false, calcDs = true)
    nS = length(lamSs)
    lambdas = Array{Float64, 2}(nS*nB, 2)
    Fits = Array{Float64, 1}(nS*nB)
    lambdasi = 1
    @showprogress for Si = 1:nS
        lamS = lamSs[(nS+1) - Si]
        #Note: lamS <= lamB <= ntasks*lamS
        lamBs = logspace(log10(lamS), log10(2*lamS), nB)
        for Bi = 1:nB
            lamB = lamBs[(nB + 1) - Bi]
            edge_confs, edge_signs = GetBestNets(Xs, YSs,
                lamS, lamB; priors = priors, ntasks = ntasks)
            sparseNets = Array{Array{Any, 2},1}(ntasks)
            for k = 1:ntasks
                filePathName = ""
                Nets = [edge_confs[k],reshape(tiedrank(edge_confs[k][:]), size(edge_confs[k])),
                    edge_signs[k]]
                headerNames = ["TF" "Gene" "Variance_Explained" "Rank" "Sign"]
                sparseNets[k] = buildNetOutputs(Nets, geneNames, TFnames,
                    headerNames)[2:end,:]
            end
            AUPR, MCC, F1,~,~ = getComparison(sparseNets, gs, gsTargsString,
            geneNames, TFnames, rankCol; extrapolation = extrapolation, makePlot = false)
            if measure == :AUPR;currFit = mean(AUPR)
            elseif measure == :MCC;currFit = mean(MCC)
            elseif measure == :F1 || throw(ArgumentError("measure not supported"))
                currFit = mean(F1)
            end
            Fits[lambdasi] = -currFit
            lambdas[lambdasi,:] = [lamS,lamB]
            lambdasi += 1
        end
    end
    networkFits = Fits
    lambda_sum = lambdas[:,1] + lambdas[:,2]/2
    sorted_indexes = sortperm(lambda_sum)
    sorted_lambda = lambdas[sorted_indexes,:]
    sortedFits = networkFits[sorted_indexes]
    chosenInd = within1seMin(sortedFits)
    chosenLams = sorted_lambda[chosenInd,:]
    minInd = indmin(networkFits)
    p::Plots.Plot = plot(lambdas[:,1],lambdas[:,2], seriestype=:scatter,
        marker_z = networkFits, markershape = :rect, markersize = 6,
        seriescolor = :Spectral, xlabel = "lambda S", ylabel = "lambda B",
        label = "", colorbar_title = "Fit", xscale = :log10, yscale = :log10,
        xlims = (10^(log10(minimum(lambdas[:,1]))-0.1), 10^(log10(maximum(lambdas[:,1]))+0.1)),
        ylims = (10^(log10(minimum(lambdas[:,2]))-0.1), 10^(log10(maximum(lambdas[:,2]))+0.1)),
        annotations = [(chosenLams[1], chosenLams[2], text("o", :darkorange)),
        (lambdas[minInd,1], lambdas[minInd,2], text("x", :violet))])
    return chosenLams, lambdas, networkFits, p
end


function getTaskNetworks(Xs::Array{Array{Float64,2},1}, YSs::Array{Array{Float64,2},1},
    priors::Array{Array{Float64,2},1}, lamS::Float64, lamB::Float64, TaskNames::Array{String, 1},
    nboots::Int64, targetGenes::Array{String,1}, targetTFs::Array{String,1})
    """Given optimal lambda pair, return our optimal network"""
    ntasks = length(TaskNames)
    if nboots == 1
        bootstrap = false
    else
        bootstrap = true
    end
    confs = Array{Array{Array{Float64,2},1},1}(ntasks)
    ranks = Array{Array{Array{Float64,2},1},1}(ntasks)
    signs = Array{Array{Array{Float64,2},1},1}(ntasks)
    for task = 1:ntasks
        confs[task] = Array{Array{Float64,2},1}(nboots)
        ranks[task] = Array{Array{Float64,2},1}(nboots)
        signs[task] = Array{Array{Float64,2},1}(nboots)
    end
    println("Getting bootstrap confidences for "*string(nboots)*" bootstraps")
    @showprogress for boot = 1:nboots
        currConfs, currSigns = GetBestNets(Xs, YSs,
        lamS, lamB, priors = priors, ntasks = ntasks, bootstrap = bootstrap)
        for task = 1:ntasks
            confs[task][boot] = currConfs[task]
            ranks[task][boot] = reshape(tiedrank(currConfs[task][:]), size(currConfs[task]))
            signs[task][boot] = currSigns[task]
        end
    end
    confsNet = Array{Array{Float64,2},1}(ntasks)
    ranksNet = Array{Array{Float64,2},1}(ntasks)
    signNet = Array{Array{Float64,2},1}(ntasks)
    countsNet = Array{Array{Float64,2},1}(ntasks)
    plots = Array{Plots.Plot{Plots.PyPlotBackend},1}(ntasks)
    for task = 1:ntasks
        currMeanConfs = mean(confs[task])
        currMeanRanks = mean(ranks[task])
        currMeanSigns = sign.(mean(signs[task]))
        currCounts = sum(map((x)->abs.(x),signs[task]))
        TFsPerGene = sum(sign.(currMeanConfs),1)[:]
        plots[task] = histogram(TFsPerGene, label = "", title = "Task: "*string(TaskNames[task]),
            xlabel = "TFs/Gene", ylabel = "count")
        confsNet[task] = currMeanConfs
        ranksNet[task] = currMeanRanks
        signNet[task] = currMeanSigns
        countsNet[task] = currCounts
    end
    p::Plots.Plot = plot(plots[:]..., layout = ntasks)
    display(p)
    sparseNets = Array{Array{Any, 2},1}(ntasks)
    for k = 1:ntasks
        tname = TaskNames[k]
        Nets = [confsNet[k],ranksNet[k],countsNet[k],signNet[k]]
        headerNames = ["TF" "Gene" "Variance_Explained" "Rank" "Count" "Sign"]
        sparseNets[k] = buildNetOutputs(Nets, targetGenes, targetTFs,
            headerNames)
    end
    return sparseNets, p
end
