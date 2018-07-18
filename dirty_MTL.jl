#Peter DeWeirdt
#07/13/2018

#Pkg.add("Plots")
#Pkg.add("StatPlots")
#Pkg.add("MAT")
#Pkg.add("OhMyREPL")
#Pkg.add("MultivariateStats")
using OhMyREPL
using StatsBase
using Plots
using ProgressMeter
using StatPlots
using MAT
using IterTools
using MultivariateStats
using DataFrames
pyplot()

function preprocess_data(Xs::Array{Array{Float64,2},1}, YSs::Array{Array{Float64,2},1})
    """z-score data"""
    ntasks = length(Xs)
    for k in 1:ntasks
        X = Xs[k]
        Ys = YSs[k]
        scoredX = zscore(X, mean(X,1), std(X, 1, corrected = true))
        scoredYs = zscore(Ys, mean(Ys,1), std(Ys,1, corrected = true))
        #If our data does not vary, set NaN's to 0. May want to set a warning for this later
        scoredX[isnan.(scoredX)] = 0
        scoredYs[isnan.(scoredYs)] = 0
        Xs[k] = scoredX
        YSs[k] = scoredYs
    end
    return Xs, YSs
end

function covariance_update_terms(Xs::Array{Array{Float64,2},1}, Ys::Array{Array{Float64,1},1};
    calcCs = true, calcDs = true)
    """Returns C and D, covariance update terms for OLS fit
    C: t(X)*Y -- correlation between predictors and response
    D: t(X)*X -- correlation between predictors and predictors
    ref: Friedman, Hastie, Tibshirani, 2010 in Journal of Statistical Software
        Regularization Paths for Generalized Linear Models via Coordinate Descent."""
    ntasks = length(Xs)
    Cs = Array{Array{Float64,1},1}(ntasks)
    Ds = Array{Array{Float64,2},1}(ntasks)
    for k in 1:ntasks
        if calcCs
            Y = Ys[k]
            X = Xs[k]
            Cs[k] = X'*Y
        end
        if calcDs
            X = Xs[k]
            Ds[k] = X'*X
        end
    end
    return Cs, Ds
end

function updateS(Cs::Array{Array{Float64,1},1}, Ds::Array{Array{Float64,2},1},
    B::Array{Float64,2}, S::Array{Float64,2}, lamS::Float64, P::Array{Float64,2})
    """returns updated coefficients for S-- (sparse matrix: predictors x tasks)
    lasso regularized -- using cyclical coordinate descent and
    soft-thresholding"""
    ntasks = length(Cs)
    npreds = size(P,1)
    for k = 1:ntasks
        c = Cs[k]; D = Ds[k]
        b = B[:,k]; s = S[:,k]
        p = P[:,k];
        for j = 1:npreds
            #if the predictor is not predictive
            s_j_zero = s
            s_j_zero[j] = 0
            if D[j,j] == 0
                temp_sj = 0
            #else update
            else
                w_temp = b .+ s_j_zero #devectorize for the sake of speed
                pred_j_corrs = D[j,:]
                temp_sj = (c[j] - sum(w_temp' * pred_j_corrs))/D[j,j]
            end
            #regularization
            if (abs(temp_sj) <= p[j]*lamS)
                s[j] = 0
            else
                s[j] = temp_sj - sign(temp_sj)*p[j]*lamS
            end
        end
        S[:,k] = s
    end
    return S
end

function updateB(Cs::Array{Array{Float64,1},1}, Ds::Array{Array{Float64,2},1},
    B::Array{Float64,2}, S::Array{Float64,2}, lamB::Float64)
    """returns updated coefficients for B (predictors x tasks)
        block regularized (l_1/l_inf) -- using cyclical coordinate descent and
        soft-thresholding on the l_1 norm across tasks
        reference: Liu et al, ICML 2009. Blockwise coordinate descent procedures
        for the multi-task lasso, with applications to neural semantic basis discovery."""
    ntasks = length(Cs)
    npreds = size(B, 1)
    #cycle through predictors
    for j = 1:npreds
        weights = zeros(ntasks)
        for k = 1:ntasks
            c = Cs[k]; D = Ds[k]
            b = B[:,k]; s = S[:,k]
            b_j_zero = b
            b_j_zero[j] = 0
            #if the predictor is not predictive
            if D[j,j] == 0
                weights[k] = 0
            #else update
            else
                w_temp = b_j_zero .+ s #devectorize for the sake of speed
                pred_j_corrs = D[j,:]
                weights[k] = (c[j] - sum(w_temp' * pred_j_corrs))/D[j,j]
            end
        end
        #set all tasks to zero if their l1-norm is too small
        if (sum(abs.(weights)) <= lamB)
            B[j,:] = 0
        else
            #Find number of coefs that would make l1-norm > lamB
            sorted_i = sortperm(weights, rev = true)
            sorted_weights = weights[sorted_i]
            m_star = 0
            f_max = 0
            for mi = 1:ntasks
                f_val = (sum(abs.(sorted_weights[1:mi])) - lamB)/mi
                if (f_val > f_max)
                    m_star = mi
                    f_max = f_val
                end
            end
            new_weights = zeros(size(weights))
            for innerk = 1:ntasks
                idx = sorted_i[innerk]
                #keep if wont put us over
                if innerk > m_star
                    new_weights[idx] = sorted_weights[innerk]
                #else make coeff the average of the others. Forces similarity
                else
                    new_weights[idx] = sign(sorted_weights[innerk])*f_max
                end
            end
            B[j,:] = new_weights
        end
    end
    return B
end

function dirty_multitask_lasso(Xs::Array{Array{Float64,2},1}, Ys::Array{Array{Float64,1},1};
    P = nothing , lamB = 0., lamS = 0.,
    Cs = nothing, Ds = nothing, S = nothing, B = nothing, maxiter = 1000,
    tolerance = 1e-4, score = false, ntasks = nothing, npreds = nothing)
    """Fits regression model in which the weights matrix W (predictors x tasks)
    is decomposed in two components: B that captures block structure across tasks
    and S that allows for the differences.
    reference: Jalali et al., NIPS 2010. A Dirty Model for Multi-task Learning."""
    if ntasks == nothing;ntasks = length(Xs);end
    if npreds == nothing;npreds = size(Xs[1], 2);end
    if score
        Xs, Ys = preprocess_data(Xs, Ys)
    end
    if P == nothing
        P = ones(npreds, ntasks)
    end
    if Cs == nothing || Ds == nothing
        Cs, Ds = covariance_update_terms(Xs, Ys)
    end
    if S == nothing;S = zeros(npreds, ntasks);end
    if B == nothing;B = zeros(npreds, ntasks);end
    W = S .+ B
    SOut = zeros(npreds, ntasks)
    BOut = zeros(npreds, ntasks)
    currDs = deepcopy(Ds)
    currCs = deepcopy(Cs)
    original_indices = 1:npreds
    for i = 1:maxiter
        W_old = deepcopy(W)
        S = updateS(currCs, currDs, B, S, lamS, P)
        B = updateB(currCs, currDs, B, S, lamB)
        active_set = find(maximum(S.+B,2) .!= 0)
        S = S[active_set,:]
        B = B[active_set,:]
        P = P[active_set,:]
        for task in 1:ntasks
            currCs[task] = currCs[task][active_set]
            currDs[task] = currDs[task][active_set,active_set]
        end
        original_indices = original_indices[active_set]
        W[setdiff(1:end,original_indices),:] = 0.
        W[original_indices,:] = S .+ B
        if maximum(abs.(W-W_old)) < tolerance
            break
        end
        if i == maxiter
            println("Maxed out on iterations for one fit, maximum parameter difference wass: "*string(maximum(abs.(W-W_old))))
        end
    end
    SOut[original_indices,:] = S
    BOut[original_indices,:] = B
    return(W, BOut, SOut)
 end

function get_RSS(X, Y, W)
    return(sum((Y - X*W).^2))
end

function approxLnNFact(n::Int64)
    # Use the sterling approximation to get ln(n!)
    return((n*log(n) - n + 0.5*log(2*pi*n)))
end

function ebic(Xs::Array{Array{Float64,2},1}, Ys::Array{Array{Float64,1},1},
    W::Array{Float64,2}, n_tasks::Int64, n_samples::Array{Int64,1}, n_preds::Int64;
    gamma = 1)
    """Calculate EBIC for each task and take the mean"""
    EBIC = Array{Float64,1}(n_tasks)
    for k in 1:n_tasks
        samps = n_samples[k]
        currW = W[:,k]
        tot_preds = length(currW)
        nonzero_pred = sum(Int,sum(abs.(currW) .> 0))
        RSS = get_RSS(Xs[k],Ys[k],W[:,k])
        BIC_penalty = nonzero_pred*log(samps)
        if nonzero_pred == 0 || nonzero_pred == tot_preds
            EBIC_penalty = 0
        else
            EBIC_penalty = 2*gamma*(approxLnNFact(tot_preds) -
            (approxLnNFact(nonzero_pred) + approxLnNFact((tot_preds - nonzero_pred))))
        end
        EBIC[k] = (samps*log(RSS/samps) + BIC_penalty + EBIC_penalty)
    end
    return(mean(EBIC))
end

function kfoldperm(N::Int64,k::Int64)
    n,r = divrem(N,k)
    b = collect(1:n:N+1)
    for i in 1:length(b)
        b[i] += i > r ? r : i-1
    end
    p = randperm(N)
    return [p[r] for r in [b[i]:b[i+1]-1 for i=1:k]]
end

function fit_gene_cv(foldInXs::Array{Array{Array{Float64,2},1},1},
    foldInYs::Array{Array{Array{Float64,1},1},1}, foldLOXs::Array{Array{Array{Float64,2},1},1},
    foldLOYs::Array{Array{Array{Float64,1},1},1}, foldDs::Array{Array{Array{Float64,2},1},1},
    ntasks::Int64, npreds::Int64,nsamps::Array{Int64,1}, lamSs::Array{Float64,1};
    nB = 3, prior = nothing)
    """For one gene, calculate fit for each pair of lamS, lamB using cross validation
    Note: sliding window for lambdaB"""
    nfolds = length(foldInYs)
    foldCs = Array{Array{Array{Float64,1},1},1}(nfolds)
    for fold = 1:nfolds
        foldCs[fold],~ = covariance_update_terms(foldInXs[fold], foldInYs[fold],
            calcDs = false, calcCs = true)
    end
    nS = length(lamSs)
    lambdas = Array{Float64, 2}(nS*nB, 2)
    Fits = Array{Float64, 1}(nS*nB)
    #Use a warm start
    lambdasi = 1
    outerS = nothing
    outerB = nothing
    for Si = 1:nS
        lamS = lamSs[(nS+1) - Si]
        #Note: lamS <= lamB <= ntasks*lamS
        lamBs = logspace(log10(lamS), log10(2*lamS), nB)
        S = outerS
        B = outerB
        for Bi = 1:nB
            lamB = lamBs[(nB + 1) - Bi]
            foldErrors = Array{Float64, 1}(nfolds)
            for fold = 1:nfolds
                W,B,S = dirty_multitask_lasso(foldInXs[fold], foldInYs[fold];
                    P = prior, lamB = lamB, lamS = lamS,
                    S = S, B = B, ntasks = ntasks, npreds = npreds)
                sq_err = 0
                for k = 1:ntasks
                    sq_err += get_RSS(foldLOXs[fold][k],foldLOYs[fold][k],W[:,k])
                end
                foldErrors[fold] = sq_err
            end
            currFit = mean(foldErrors)
            if Bi == 1
                outerS = S
                outerB = B
            end
            Fits[lambdasi] = currFit
            lambdas[lambdasi,:] = [lamS,lamB]
            lambdasi += 1
        end
    end
    return(Fits, lambdas)
end

function fit_gene_ebic(Xs::Array{Array{Float64,2},1}, Ys::Array{Array{Float64,1},1},
    Ds::Array{Array{Float64,2},1}, ntasks::Int64, npreds::Int64,nsamps::Array{Int64,1},
    lamSs::Array{Float64,1}; nB = 3, prior = nothing)
    """For one gene, calculate fit for each pair of lamS, lamB
    fit options
    Note: sliding window for lambdaB"""
    Cs,~ = covariance_update_terms(Xs, Ys, calcDs = false, calcCs = true)
    nS = length(lamSs)
    lambdas = Array{Float64, 2}(nS*nB, 2)
    Fits = Array{Float64, 1}(nS*nB)
    #Use a warm start
    lambdasi = 1
    outerS = nothing
    outerB = nothing
    for Si = 1:nS
        lamS = lamSs[(nS+1) - Si]
        #Note: lamS <= lamB <= ntasks*lamS
        lamBs = logspace(log10(lamS), log10(2*lamS), nB)
        S = outerS
        B = outerB
        for Bi = 1:nB
            lamB = lamBs[(nB + 1) - Bi]
            W,B,S = dirty_multitask_lasso(Xs, Ys;
                P = prior, lamB = lamB, lamS = lamS,
                Cs = Cs, Ds = Ds, S = S, B = B, ntasks = ntasks, npreds = npreds)
            currFit = ebic(Xs, Ys, W, ntasks, nsamps, npreds)
            if Bi == 1
                outerS = S
                outerB = B
            end
            Fits[lambdasi] = currFit
            lambdas[lambdasi,:] = [lamS,lamB]
            lambdasi += 1
        end
    end
    return(Fits, lambdas)
end

function within1seMin(data::Array{Float64,1})
    #= Get Largest lambda lambda value within 1se of the minimum =#
    se = std(data)/sqrt(length(data))
    minval, minind = findmin(data)
    maxval = minval + se
    closeEnough = find(data .< maxval);
    ind1se = maximum(closeEnough[closeEnough .>= minind])
    return(ind1se)
end

function fit_network(Xs::Array{Array{Float64,2},1},
    YSs::Array{Array{Float64,2},1}; Smin = 0.01, Smax = 1,
    Ssteps = 10, nB = 4, priors = nothing, fit = :ebic, npreds = nothing,
    nsamps = nothing, ngenes = nothing, nfolds = 5)
    """Calculate fit for each gene in a network. Plot grid of median fits.
    Return lambdaS and lambdaB within 1 se of the minimum fit."""
    ntasks = length(Xs)
    if npreds==nothing;npreds = size(Xs[1], 2);end
    if nsamps==nothing
        nsamps = Array{Int64, 1}(ntasks)
        for task = 1:ntasks
            nsamps[task] = size(Xs[task], 1)
        end
    end
    if ngenes==nothing;ngenes = size(YSs[1],2);end
    lamSlog10step = 1/Ssteps
    logLamSRange = log10(Smin):lamSlog10step:log10(Smax)
    lamSs = 10.^logLamSRange
    geneFits = Array{Float64,2}(length(lamSs)*nB,ngenes)
    lambdas = Array{Float64,2}(length(lamSs)*nB,2)
    if fit == :ebic
        Xs,YSs = preprocess_data(Xs, YSs)
        ~,Ds = covariance_update_terms(Xs, Array{Array{Float64,1},1}(0), calcCs = false, calcDs = true)
    elseif fit == :cv
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
    end
    println("Estimating fits for " * string(ngenes) * " genes")
    @showprogress for genei = 1:ngenes
        if fit == :ebic
            Ys = Array{Array{Float64,1},1}(ntasks)
            P = Array{Float64,2}(npreds, ntasks)
            for k = 1:ntasks
                Ys[k] = YSs[k][:,genei]
                if priors != nothing
                    P[:,k] = priors[k][:,genei]
                else
                    P[:,k] = ones(npreds)
                end
            end
            Fits, lambdas = fit_gene_ebic(Xs, Ys, Ds,
            ntasks, npreds, nsamps, lamSs, nB = nB, prior = P)
        elseif fit == :cv
            P = Array{Float64,2}(npreds, ntasks)
            for k = 1:ntasks
                if priors != nothing
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
        end
        geneFits[:,genei] = Fits
    end
    networkFits = median(geneFits,2)
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
    display(p)
    fitsOut = "EstimatedFits_" * string(fit)
    savefig(fitsOut * ".pdf")
    matwrite(fitsOut * ".mat", Dict(
            "lambdas" => lambdas,
            "geneFits" => geneFits,
            "networkFits" => networkFits
    ))
    return chosenLams
end

function GetBestNets(Xs::Array{Array{Float64,2},1}, YSs::Array{Array{Float64,2},1},
    lamS::Float64, lamB::Float64; priors = nothing, ntasks = nothing, npreds = nothing,
    ngenes = nothing)
    """ Given an optimal lamS and lamB return a matrix of confidences for edge
    interactions, and the sign of those interactions. Current options for methods
    are confidences and ranks"""
    Xs, YSs = preprocess_data(Xs, YSs)
    if ntasks == nothing;ntasks = length(Xs);end
    ~,Ds = covariance_update_terms(Xs, Array{Array{Float64,1},1}(0), calcCs = false, calcDs = true)
    if npreds == nothing;npreds = size(Xs[1], 2);end
    if ngenes == nothing;ngenes = length(YSs);end
    edge_confs = Array{Array{Float64,2},1}(ntasks)
    edge_ranks = Array{Array{Float64,2},1}(ntasks)
    edge_signs = Array{Array{Float64,2},1}(ntasks)
    for task = 1:ntasks
        edge_confs[task] = zeros(npreds, ngenes)
        edge_ranks[task] = zeros(npreds, ngenes)
        edge_signs[task] = zeros(npreds, ngenes)
    end
    for genei = 1:ngenes
        Ys = Array{Array{Float64,1},1}(ntasks)
        P = Array{Float64,2}(npreds, ntasks)
        for k = 1:ntasks
            Ys[k] = YSs[k][:,genei]
            if priors != nothing
                P[:,k] = priors[k][:,genei]
            else
                P[:,k] = ones(npreds)
            end
        end
        Cs,~ = covariance_update_terms(Xs, Ys, calcDs = false, calcCs = true)
        W,B,S = dirty_multitask_lasso(Xs, Ys;
            P = P, lamB = lamB, lamS = lamS,
            Cs = Cs, Ds = Ds, ntasks = ntasks, npreds = npreds)
        for task = 1:ntasks
            currBeta = W[:,task]
            currYs = Ys[task]
            nzeroPreds = find(currBeta)
            nsamps = size(Xs[task],1)
            if length(nzeroPreds) >  nsamps #Might want to print a warning for this too
                sortedIndexes = sortperm(currBeta[nzeroPreds], rev = true)
                nzeroPreds = nzeroPreds[sortedIndexes][1:(nsamps-1)]
            end
            currNzeroXs = Xs[task][:,nzeroPreds]
            rescaledBeta = llsq(currNzeroXs, currYs, bias = false)
            varResidAll = var(currNzeroXs*rescaledBeta - currYs)
            gene_edge_confs = zeros(npreds)
            for nzeroPred = 1:length(nzeroPreds)
                ogIndex = nzeroPreds[nzeroPred]
                noPredBeta = deepcopy(rescaledBeta)
                noPredBeta[nzeroPred] = 0
                varResidPred = var(currNzeroXs*noPredBeta - currYs)
                gene_edge_confs[ogIndex] = (1-(varResidAll/varResidPred))
            end
            edge_confs[task][:,genei] = gene_edge_confs
            gene_edge_ranks = tiedrank(gene_edge_confs,rev = true)
            edge_ranks[task][:,genei] = gene_edge_ranks
            currEdgeSigns = zeros(npreds)
            currEdgeSigns[nzeroPreds] = sign.(rescaledBeta)
            edge_signs[task][:,genei] = currEdgeSigns
        end
    end
    return edge_confs, edge_ranks, edge_signs
end

function get_partitions(data::Array{Float64,1}, nsamps::Array{Int64,1})
    #Assume length(data) %% nparts = 0
    nparts = length(nsamps)
    partitionedData = Array{Array{Any,1},1}(nparts)
    for part = 1:nparts
        partitionedData[part] = data[(sum(nsamps[1:(part-1)])+1):sum(nsamps[1:part])]
    end
    return(partitionedData)
end

function buildTRNs(Xs::Array{Array{Float64,2},1}, YSs::Array{Array{Float64,2},1};
    Smin = 0.05, Smax = 1, Ssteps = 10, nB = 4, nboots = 1, priors = nothing,
    fit = :ebic, nfolds = 2, nsamps = nothing, ntasks = nothing)
    """Rank TF-gene interactions according to confidence:
        1 - var(residuals_i)/var(residuals[~TF]).
    Use rankings to return a transcritptional regulatory network."""
    if ntasks == nothing;ntasks = lenght(Xs);end
    if nsamps == nothing
        for k = 1:ntasks
            nsamps[k] = size(Xs[k],1)
        end
    end
    confs = Array{Array{Array{Float64,2},1},1}(ntasks)
    ranks = Array{Array{Array{Float64,2},1},1}(ntasks)
    signs = Array{Array{Array{Float64,2},1},1}(ntasks)
    for task = 1:ntasks
        confs[task] = Array{Array{Float64,2},1}(nboots)
        ranks[task] = Array{Array{Float64,2},1}(nboots)
        signs[task] = Array{Array{Float64,2},1}(nboots)
    end
    for boot = 1:nboots
        println("Getting bootstrap confidences for "*string(boot)*" bootstraps")
        sampleXs = deepcopy(Xs)
        sampleYSs = deepcopy(YSs)
        if nboots != 1
            for task = 1:ntasks
                currSamps = nsamps[task]
                samples = sample(1:currSamps,currSamps)
                sampleXs[task] =  Xs[task][samples,:]
                sampleYSs[task] = YSs[task][samples,:]
            end
        end
        lams =  fit_network(sampleXs, sampleYSs, Smin = Smin,
        Smax = Smax, Ssteps = Ssteps, nB = nB, priors = priors, fit = fit, nfolds = nfolds)
        println(size(lams))
        lamS = lams[1]; lamB = lams[2]
        currConfs, currRanks, currSigns = GetBestNets(Xs, YSs,
        lamS, lamB, priors = priors, ntasks = ntasks)
        for task = 1:ntasks
            confs[task][boot] = currConfs[task]
            ranks[task][boot] = currRanks[task]
            signs[task][boot] = currSigns[task]
        end
    end
    confsNet = Array{Array{Float64,2},1}(ntasks)
    ranksNet = Array{Array{Float64,2},1}(ntasks)
    signNet = Array{Array{Float64,2},1}(ntasks)
    plots = Array{Plots.Plot{Plots.PyPlotBackend},1}(ntasks)
    for task = 1:ntasks
        currMeanConfs = mean(confs[task])
        currMeanRanks = mean(ranks[task])
        currMeanSigns = mean(signs[task])
        TFsPerGene = sum(sign.(currMeanConfs),1)
        plots[task] = histogram(TFsPerGene, label = "", title = "Task: "*string(task),
            xlabel = "TFs/Gene", ylabel = "count")
        confsNet[task] = currMeanConfs
        ranksNet[task] = currMeanRanks
        signNet[task] = currMeanSigns
    end
    p::Plots.Plot = plot(plots[:]..., layout = ntasks)
    display(p)
    TRNOut = "TRNs"*string(fit)
    savefig(TRNOut * ".pdf")
    matwrite(TRNOut * ".mat", Dict(
            "Confs" => confsNet,
            "Ranks" => ranksNet,
            "Signs" => signNet
    ))
    return(confsNet, ranksNet, signNet)
end

function buildOutputs(confsNet::Array{Float64,2}, targetGenes::Array{String,1},
    targetTFs::Array{String,1}; otherNets = nothing)
    """Output a sparse tsv of network connections. Must have a confidence network
    in which rows are TFs and columns are genes"""
    nzero = find(confsNet .!= 0)
    nzeroRows_Cols = ind2sub(size(confsNet),nzero)
    nzeroTFs = targetTFs[nzeroRows_Cols[1]]
    nzeroGenes = targetGenes[nzeroRows_Cols[2]]
    nzeroConfs = confsNet[nzero]
    sparseNet = hcat(nzeroTFs, nzeroGenes, nzeroConfs)
    sortedSparseNet = sortrows(sparseNet, by = (x)->x[3], rev = true)
    return sortedSparseNet
end

function getAUPR(confsNets::Array{Array{Float64, 2},1}, gs::Array{String,2},
    geneNames::Array{String, 1}, TFnames::Array{String, 1})
    ntasks = length(confsNets)
    AUPRs = Array{Float64, 1}(ntasks)
    for k = 1:ntasks
        confsNet = confsNets[k]
        sortedSparseNet = buildOutputs(confsNet, geneNames, TFnames)
        validInteractions = intersect(find(indexin(sortedSparseNet[:,1],gs[:,1])),
            find(indexin(sortedSparseNet[:,2],gs[:,2])))
        validNet = sortedSparseNet[validInteractions,:]
        nInferred = size(validNet,1)
        infTuples = Array{Tuple{String,String},1}(nInferred)
        for i = 1:nInferred
            infTuples[i] = (validNet[i,[1,2]]...)
        end
        TEdges = size(gs, 2)
        gsTuples = Array{Tuple{String,String},1}(TEdges)
        for i = 1:TEdges
            gsTuples[i] = (gs[i,:]...)
        end
        InGs = sign.(indexin(infTuples, gsTuples))
        totLevels = length(InGs)
        precisions = zeros(totLevels)
        recalls = zeros(totLevels)
        for lev = 1:totLevels
            numTrue = sum(InGs[1:lev])
            precisions[lev] = numTrue/lev
            recalls[lev] = numTrue/TEdges
        end
        heights = (precisions[2:end] + precisions[1:end - 1])/2
        widths = recalls[2:end] - recalls[1:end-1]
        AUPRs[k] = heights' * widths
    end
    return -(mean(AUPRs))
end

function fit_network_AUPR(Xs::Array{Array{Float64,2},1},
    YSs::Array{Array{Float64,2},1}, gs::Array{String,2}, geneNames::Array{String,1},
    TFnames::Array{String,1}; Smin = 0.01, Smax = 1,
    Ssteps = 10, nB = 4, priors = nothing, fit = :AUPR, npreds = nothing,
    nsamps = nothing, ngenes = nothing)
    """Calculate fit for each gene in a network. Plot grid of median fits.
    Return lambdaS and lambdaB within 1 se of the minimum fit."""
    ntasks = length(Xs)
    if npreds==nothing;npreds = size(Xs[1], 2);end
    if nsamps==nothing
        nsamps = Array{Int64, 1}(ntasks)
        for task = 1:ntasks
            nsamps[task] = size(Xs[task], 1)
        end
    end
    if ngenes==nothing;ngenes = size(YSs[1],2);end
    lamSlog10step = 1/Ssteps
    logLamSRange = log10(Smin):lamSlog10step:log10(Smax)
    lamSs = 10.^logLamSRange
    geneFits = Array{Float64,2}(length(lamSs)*nB,ngenes)
    lambdas = Array{Float64,2}(length(lamSs)*nB,2)
    Xs,YSs = preprocess_data(Xs, YSs)
    ~,Ds = covariance_update_terms(Xs, Array{Array{Float64,1},1}(0), calcCs = false, calcDs = true)
    nS = length(lamSs)
    lambdas = Array{Float64, 2}(nS*nB, 2)
    Fits = Array{Float64, 1}(nS*nB)
    #Use a warm start
    lambdasi = 1
    outerS = nothing
    outerB = nothing
    println("Getting AUPR for each lambda pair")
    @showprogress for Si = 1:nS
        lamS = lamSs[(nS+1) - Si]
        #Note: lamS <= lamB <= ntasks*lamS
        lamBs = logspace(log10(lamS), log10(2*lamS), nB)
        S = outerS
        B = outerB
        for Bi = 1:nB
            lamB = lamBs[(nB + 1) - Bi]
            edge_confs, edge_ranks, edge_signs = GetBestNets(Xs, YSs,
                lamS, lamB; priors = priors, ntasks = ntasks, npreds = npreds,
                ngenes = ngenes)
            currFit = getAUPR(edge_confs, gs, geneNames, TFnames)
            if Bi == 1
                outerS = S
                outerB = B
            end
            Fits[lambdasi] = currFit
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
    display(p)
    fitsOut = "EstimatedFits_" * string(fit)
    savefig(fitsOut * ".pdf")
    matwrite(fitsOut * ".mat", Dict(
            "lambdas" => lambdas,
            "geneFits" => geneFits,
            "networkFits" => networkFits
    ))
    return chosenLams
end

function getGSfits(Xs::Array{Array{Float64,2},1}, YSs::Array{Array{Float64,2},1},
    gs::Array{String,2}, geneNames::Array{String,1}, TFnames::Array{String,1};
    Smin = 0.05, Smax = 1, Ssteps = 10, nB = 3, nboots = 1, priors = nothing,
    fit = :ebic, nsamps = nothing, ntasks = nothing)
    if ntasks == nothing;ntasks = lenght(Xs);end
    if nsamps == nothing
        for k = 1:ntasks
            nsamps[k] = size(Xs[k],1)
        end
    end
    println("Getting AUPR")
    sampleXs = deepcopy(Xs)
    sampleYSs = deepcopy(YSs)
    if nboots != 1
        for task = 1:ntasks
            currSamps = nsamps[task]
            samples = sample(1:currSamps,currSamps)
            sampleXs[task] =  Xs[task][samples,:]
            sampleYSs[task] = YSs[task][samples,:]
        end
    end
    lams =  fit_network_AUPR(sampleXs, sampleYSs, gs, geneNames, TFnames, Smin = Smin,
        Smax = Smax, Ssteps = Ssteps, nB = nB, priors = priors)
    return(lams)
end


function MTL(nboots = 1)
    """MAIN: fit TRNs for multiple tasks using multitask learning"""
    TaskMatPaths = ["./fitSetup/RNAseqWmicro20genes_ATAC_Th17_bias50.mat",
        "./fitSetup/microarrayWbulk20genes_ATAC_Th17_bias50.mat"]
    gs = readdlm("micro_RNAseq_small_GS.txt",String)
    ntasks = length(TaskMatPaths)
    nsamps = Array{Int64}(ntasks)
    ngenes = Int64
    nTFs = Int64
    taskMTLinputs = Array{Dict{String,Any},1}(ntasks)
    Xs = Array{Array{Float64,2},1}(ntasks)
    YSs = Array{Array{Float64,2},1}(ntasks)
    priors = Array{Array{Float64,2},1}(ntasks)
    println("Reading in .mat data for MTL inference")
    @showprogress for task = 1:ntasks
        inputs = matread(TaskMatPaths[task])
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
    println("Buliding TRNs using ebic")
    confsNet, ranksNet, signNet = buildTRNs(Xs, YSs,
        Smin = 0.02, Smax = 1, Ssteps = 10, nB = 3, nboots = 1, priors = priors,
        fit = :ebic, nsamps = nsamps, ntasks = ntasks)
    println("EBIC AUPR: ",getAUPR(confsNet, gs, geneNames, TFNames))
    println("Buliding TRNs using cv")
    confsNet, ranksNet, signNet = buildTRNs(Xs, YSs,
        Smin = 0.02, Smax = 1, Ssteps = 10, nB = 3, nboots = 1, priors = priors,
        fit = :cv, nsamps = nsamps, ntasks = ntasks, nfolds = 2)
    println("CV AUPR: ",getAUPR(confsNet, gs, geneNames, TFNames))
    getGSfits(Xs, YSs, gs, geneNames, TFNames,
        Smin = 0.02, Smax = 1, Ssteps = 10, nB = 3, nboots = 1, priors = priors,
        fit = :AUPR, nsamps = nsamps, ntasks = ntasks)
end
