#=
dirty_MTL.jl
A Dirty Approach to Multitask Learning written in Julia
Author: Peter DeWeirdt, Divisions of Immunobiology and Biomedical Informatics,
    Cincinnati Children's Hospital
References:
(1) Jalali et al. (2010) "A dirty model for multi-task learning."
(2) Castro, De Veaux, Miraldi, Bonneau. (2018) "Multitask learning for joint inference of gene
    regulatory networks from several expressoin datasets"
Includes multitask learning algorithm. 
=#
include("dependencies.jl")

function preprocess_data(Xs::Array{Array{Float64,2},1}, YSs::Array{Array{Float64,2},1})
    """z-score data"""
    ntasks = length(Xs)
    for k in 1:ntasks
        X = Xs[k]
        Ys = YSs[k]
        scoredX = zscore(X, mean(X,1), std(X, 1, corrected = true))
        scoredYs = zscore(Ys, mean(Ys,1), std(Ys,1, corrected = true))
        #If our data does not vary, set NaN's to 0.
        scoredX[isnan.(scoredX)] = 0
        scoredYs[isnan.(scoredYs)] = 0
        Xs[k] = scoredX
        YSs[k] = scoredYs
    end
    return Xs, YSs
end

function covariance_update_terms(Xs::Array{Array{Float64,2},1}, Ys::Array{Array{Float64,1},1};
    calcCs::Bool = true, calcDs::Bool = true)
    """Returns C and D, covariance update terms for OLS fit
    C: t(X)*Y -- correlation between predictors and response
    D: t(X)*X -- correlation between predictors and predictors
    Ref: Friedman, Hastie, Tibshirani, 2010 in Journal of Statistical Software
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
    """Returns updated coefficients for S (predictors x tasks)
    lasso regularized sing cyclical coordinate descent and
    soft-thresholding Ref: Friedman, Hastie, Tibshirani, 2010 in Journal of Statistical Software
    Regularization Paths for Generalized Linear Models via Coordinate Descent."""
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
    B::Array{Float64,2}, S::Array{Float64,2}, lamB::Float64, BPrior::Array{Float64,1})
    """Returns updated coefficients for B (predictors x tasks)
        block regularized (l_1/l_inf) using cyclical coordinate descent and
        soft-thresholding on the l_1 norm across tasks
        Ref: Liu et al, ICML 2009. Blockwise coordinate descent procedures
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
            #if the predictor has variance of 0
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
        if (sum(abs.(weights)) <= lamB*BPrior[j])
            B[j,:] = 0
        else
            #Find number of coefs that would make l1-norm > lamB
            sorted_i = sortperm(weights, rev = true)
            sorted_weights = weights[sorted_i]
            m_star = 0
            f_max = 0
            for mi = 1:ntasks
                f_val = (sum(abs.(sorted_weights[1:mi])) - BPrior[j]*lamB)/mi
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
    P::Array{Float64,2} = Array{Float64,2}(0,0), lamB::Float64 = 0., lamS::Float64 = 0.,
    Cs::Array{Array{Float64, 1},1} = Array{Array{Float64, 1},1}(0), Ds::Array{Array{Float64, 2},1} = Array{Array{Float64, 2},1}(0),
    S::Array{Float64,2} = Array{Float64,2}(0,0), B::Array{Float64,2} = Array{Float64,2}(0,0), maxiter::Int64 = 10000,
    tolerance::Float64 = 1e-7, score::Bool = false, ntasks::Int64 = 0, npreds::Int64 = 0, useBlockPrior = true)
    """Fits regression model in which the weights matrix W (predictors x tasks)
    is decomposed in two components: B that captures block structure across tasks
    and S that allows for the differences.
    Ref: Jalali et al., NIPS 2010. A Dirty Model for Multi-task Learning."""
    if ntasks == 0;ntasks = length(Xs);end
    if npreds == 0;npreds = size(Xs[1], 2);end
    if score
        Xs, Ys = preprocess_data(Xs, Ys)
    end
    if length(P) == 0
        P = ones(npreds, ntasks)
    end
    if length(Cs) == 0 || length(Ds) == 0
        Cs, Ds = covariance_update_terms(Xs, Ys)
    end
    if length(S) == 0;S = zeros(npreds, ntasks);end
    if length(B) == 0;B = zeros(npreds, ntasks);end
    if useBlockPrior
        BPrior = mean(P,2)[:]
    else
        #Make block prior all 1's, except where the sparse prior has inf
        BPrior = max.(maximum(P,2), ones(size(P,1)))[:]
    end
    W = S .+ B
    SOut = zeros(npreds, ntasks)
    BOut = zeros(npreds, ntasks)
    currDs = deepcopy(Ds)
    currCs = deepcopy(Cs)
    original_indices = 1:npreds
    for i = 1:maxiter
        W_old = deepcopy(W)
        S = updateS(currCs, currDs, B, S, lamS, P)
        B = updateB(currCs, currDs, B, S, lamB, BPrior)
        active_set = find(maximum(S.+B,2) .!= 0)
        S = S[active_set,:]
        B = B[active_set,:]
        P = P[active_set,:]
        BPrior = BPrior[active_set]
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
            println("Maxed out on iterations, maximum parameter difference was: "*
                string(maximum(abs.(W-W_old))))
        end
    end
    SOut[original_indices,:] = S
    BOut[original_indices,:] = B
    return(W, BOut, SOut)
 end

function get_RSS(X, Y, W)
    """Residual sum of squares"""
    return(sum((Y - X*W).^2))
end

function approxLnNFact(n::Int64)
    """Avoid float overload by using Stirling's approximation for ln(n!)
    Ref: Wells. (1986) "The Penguin Dictionary of Curious and Interesting Numbers."""""
    return((n*log(n) - n + 0.5*log(2*pi*n)))
end

function ic(Xs::Array{Array{Float64,2},1}, Ys::Array{Array{Float64,1},1},
    W::Array{Float64,2}, n_tasks::Int64, n_samples::Array{Int64,1}, n_preds::Int64;
    gamma::Int64 = 1, fit::Symbol = :ebic, tolerance::Float64 = 1e-7,
    useBlockPrior::Bool = true)
    """Calculate the [extended] bayesian information criteria ([E]BIC) for model(s)
    and take the mean of these fits. Note: also allows for BIC fir with fit = :BIC
    Ref: Foygel, Drton. (2010) "Extended Bayesian Information Criteria for Gaussian Graphical Models" """
    EBIC = Array{Float64,1}(n_tasks)
    for k in 1:n_tasks
        samps = n_samples[k]
        currW = W[:,k]
        tot_preds = length(currW)
        currNzero = find(currW)
        nonzero_pred = length(currNzero)
        RSS = get_RSS(Xs[k],Ys[k],W[:,k])
        BIC_penalty = nonzero_pred*log(samps)
        if fit == :ebic
            if nonzero_pred == 0 || nonzero_pred == tot_preds
                EBIC_penalty = 0
            else
                EBIC_penalty = 2*gamma*(approxLnNFact(tot_preds) -
                (approxLnNFact(nonzero_pred) + approxLnNFact((tot_preds - nonzero_pred))))
            end
        elseif fit == :bic
            EBIC_penalty = 0
        else
            println("Fit method not yet supported")
        end
        EBIC[k] = (samps*log(RSS/samps) + BIC_penalty + EBIC_penalty)
    end
    return(mean(EBIC))
end

function kfoldperm(N::Int64,k::Int64)
    """Partition N samples into k folds of approximately equal sizes"""
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
    nB::Int64 = 3, prior::Array{Float64,2} = Array{Float64,2}(0,0),
    tolerance::Float64 = 1e-7, useBlockPrior::Bool = true)
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
    outerS = Array{Float64,2}(0,0)
    outerB = Array{Float64,2}(0,0)
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
                    S = S, B = B, ntasks = ntasks, npreds = npreds, tolerance = tolerance,
                    useBlockPrior = useBlockPrior)
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

function fit_gene_ic(Xs::Array{Array{Float64,2},1}, Ys::Array{Array{Float64,1},1},
    Ds::Array{Array{Float64,2},1}, ntasks::Int64, npreds::Int64,nsamps::Array{Int64,1},
    lamSs::Array{Float64,1}; nB::Int64 = 3, prior::Array{Float64,2} = Array{Float64,2}(0,0),
    fit::Symbol = :ebic, tolerance::Float64 = 1e-7, useBlockPrior::Bool = true)
    """For one gene, calculate fit for each pair of lamS, lamB
    Note: sliding window for lambdaB. Use a warm start.
    Ref: Friedman, Hastie, Tibshirani, 2010 in Journal of Statistical Software
        Regularization Paths for Generalized Linear Models via Coordinate Descent."""
    Cs,~ = covariance_update_terms(Xs, Ys, calcDs = false, calcCs = true)
    nS = length(lamSs)
    lambdas = Array{Float64, 2}(nS*nB, 2)
    Fits = Array{Float64, 1}(nS*nB)
    #Use a warm start
    lambdasi = 1
    outerS = Array{Float64,2}(0,0)
    outerB = Array{Float64,2}(0,0)
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
                Cs = Cs, Ds = Ds, S = S, B = B, ntasks = ntasks, npreds = npreds, tolerance = tolerance,
                useBlockPrior = useBlockPrior)
            currFit = ic(Xs, Ys, W, ntasks, nsamps, npreds, fit = fit)
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
    """Get Largest lambda value within 1se of the minimum"""
    se = std(data)/sqrt(length(data))
    minval, minind = findmin(data)
    maxval = minval + se
    closeEnough = find(data .<= maxval)
    ind1se = maximum(closeEnough[closeEnough .>= minind])
    return(ind1se)
end

function get_fit_outputs(geneFits::Array{Float64,2}, networkFits::Array{Float64,2},
    lambdas::Array{Float64,2}, fit::Symbol)
    """Get Optimal Lamdas and plot fits"""
    lambda_sum = lambdas[:,1] + lambdas[:,2]/2
    sorted_indexes = sortperm(lambda_sum)
    sorted_lambda = lambdas[sorted_indexes,:]
    sortedFits = networkFits[sorted_indexes]
    chosenInd = within1seMin(sortedFits)
    chosenLams = sorted_lambda[chosenInd,:]
    minInd = indmin(networkFits)
    println("Creating Plots for lambda outputs")
    p::Plots.Plot = plot(lambdas[:,1],lambdas[:,2], seriestype=:scatter,
        marker_z = networkFits, markershape = :rect, markersize = 6,
        seriescolor = :Spectral, xlabel = "lambda S", ylabel = "lambda B",
        label = "", colorbar_title = "Fit", xscale = :log10, yscale = :log10,
        xlims = (10^(log10(minimum(lambdas[:,1]))-0.1), 10^(log10(maximum(lambdas[:,1]))+0.1)),
        ylims = (10^(log10(minimum(lambdas[:,2]))-0.1), 10^(log10(maximum(lambdas[:,2]))+0.1)),
        annotations = [(chosenLams[1], chosenLams[2], text("o", :darkorange)),
        (lambdas[minInd,1], lambdas[minInd,2], text("x", :violet))])
    display(p)
    return chosenLams, p
end

function GetBestNets(Xs::Array{Array{Float64,2},1}, YSs::Array{Array{Float64,2},1},
    lamS::Float64, lamB::Float64; priors::Array{Array{Float64,2},1} = Array{Array{Float64,2},1}(0),
    ntasks::Int64 = 0, npreds::Int64 = 0, ngenes::Int64 = 0, bootstrap::Bool = false,
    tolerance::Float64 = 1e-7, useBlockPrior::Bool = true)
    """Given an optimal lamS and lamB return a matrix of confidences for edge
    interactions, and the sign of those interactions. Current options for methods
    are confidences and ranks.
    Rank TF-gene interactions according to confidence:
        1 - var(residuals_i)/var(residuals[~TF]).
    Ref: Greenfield, Hafemeister, Bonneau. (2010) "Robust data-driven incorporation of
    prior knowledge into the inference of dynamic regulatory networks." """
    if bootstrap == true
        for task = 1:ntasks
            currSamps = size(Xs[task],1)
            samples = sample(1:currSamps,currSamps)
            Xs[task] =  Xs[task][samples,:]
            YSs[task] = YSs[task][samples,:]
        end
    end
    Xs, YSs = preprocess_data(Xs, YSs)
    if ntasks == 0;ntasks = length(Xs);end
    ~,Ds = covariance_update_terms(Xs, Array{Array{Float64,1},1}(0), calcCs = false, calcDs = true)
    if npreds == 0;npreds = size(Xs[1], 2);end
    if ngenes == 0;ngenes = size(YSs[1],2);end
    edge_confs = Array{Array{Float64,2},1}(ntasks)
    edge_signs = Array{Array{Float64,2},1}(ntasks)
    for task = 1:ntasks
        edge_confs[task] = zeros(npreds, ngenes)
        edge_signs[task] = zeros(npreds, ngenes)
    end
    for genei = 1:ngenes
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
        Cs,~ = covariance_update_terms(Xs, Ys, calcDs = false, calcCs = true)
        W,B,S = dirty_multitask_lasso(Xs, Ys;
            P = P, lamB = lamB, lamS = lamS,
            Cs = Cs, Ds = Ds, ntasks = ntasks, npreds = npreds, tolerance = tolerance,
            useBlockPrior = useBlockPrior)
        for task = 1:ntasks
            currBeta = W[:,task]
            currYs = Ys[task]
            currX = Xs[task]
            nzeroPreds = find(currBeta)
            gene_edge_confs = zeros(npreds)
            currEdgeSigns = zeros(npreds)
            if length(nzeroPreds) > 0
                nsamps = size(currX,1)
                if length(nzeroPreds) >=  length(unique(currYs)) #Might want to print a warning for this too
                    sortedIndexes = sortperm(abs.(currBeta[nzeroPreds]), rev = true)
                    nzeroPreds = nzeroPreds[sortedIndexes][1:(length(unique(currYs))-1)]
                end
                currNzeroXs = currX[:,nzeroPreds]
                rescaledBeta = coef(lm(currNzeroXs, currYs, false))
                varResidAll = var(currNzeroXs*rescaledBeta - currYs)
                while isapprox(0, varResidAll; atol = 1e-20) && length(nzeroPreds) > 0
                    sortedIndexes = sortperm(abs.(rescaledBeta), rev = true)
                    nzeroPreds = nzeroPreds[sortedIndexes][1:(end-1)]
                    currNzeroXs = currX[:,nzeroPreds]
                    rescaledBeta = coef(lm(currNzeroXs, currYs, false))
                    varResidAll = var(currNzeroXs*rescaledBeta - currYs)
                end
                    for nzeroPred = 1:length(nzeroPreds)
                        ogIndex = nzeroPreds[nzeroPred]
                        noPredBeta = deepcopy(rescaledBeta)
                        noPredBeta[nzeroPred] = 0
                        varResidPred = var(currNzeroXs*noPredBeta - currYs)
                        gene_edge_confs[ogIndex] = (1-(varResidAll/varResidPred))
                    end
                    currEdgeSigns[nzeroPreds] = sign.(rescaledBeta)
            end
            edge_confs[task][:,genei] = gene_edge_confs
            edge_signs[task][:,genei] = currEdgeSigns
        end
    end
    return edge_confs, edge_signs
end

function buildNetOutputs(Nets::Array{Array{Float64,2},1}, targetGenes::Array{String,1},
    targetTFs::Array{String,1}, headerNames::Array{String, 2})
    """Output a sparse tsv of network connections. Must have a confidence network
    in which rows are TFs and columns are genes"""
    nzero = find(Nets[1] .!= 0)
    nzeroRows_Cols = ind2sub(size(Nets[1]),nzero)
    nzeroTFs = targetTFs[nzeroRows_Cols[1]]
    nzeroGenes = targetGenes[nzeroRows_Cols[2]]
    nNets = length(Nets)
    sparseNet = Array{Any, 2}(length(nzero),2+nNets)
    sparseNet[:,1] = targetTFs[nzeroRows_Cols[1]]
    sparseNet[:,2] = targetGenes[nzeroRows_Cols[2]]
    for i = 1:nNets
        sparseNet[:,2+i] = Nets[i][nzero]
    end
    sortedSparseNet = sortrows(sparseNet, by = (x)->x[3], rev = true)
    if length(headerNames) == size(sortedSparseNet,2) ||
        throw(DimensionMismatch("Size of HeaderNames not equal to ncols of sortedSparseNet"))
        NamedSparseNet = [headerNames;sortedSparseNet]
    end
    return NamedSparseNet
end

function getComparison(sparseNets::Array{Array{Any, 2},1}, gs::Array{String,2},
    geneNames::Array{String, 1}, TFnames::Array{String, 1}, rankCol::Int64,
    gsTargs::Array{String, 1}; TaskNames::Array{String,1} = Array{String,1}(0),
    extrapolation::Bool = false, makePlot::Bool = true)
    colors = [:royalblue, :indianred, :orange, :violet, :olivedrab, :cyan, :darkgoldenrod]
    ntasks = length(sparseNets)
    AUPRs = Array{Float64, 1}(ntasks)
    MCCs = Array{Float64, 1}(ntasks) # Matthews correlation coeffients
    F1s = Array{Float64, 1}(ntasks) # F1 scores
    InInfMat = Array{Any, 2}(0,(ntasks+1))
    p::Plots.Plot = plot()
    for k = 1:ntasks
        if length(TaskNames) == ntasks
            println(TaskNames[k])
        end
        sortedSparseNet = sortrows(sparseNets[k][2:end,:], by = (x)->x[rankCol], rev = true)
        validInteractions = intersect(find(indexin(sortedSparseNet[:,1],gs[:,1])),
            find(indexin(sortedSparseNet[:,2],gs[:,2])))
        validNet = sortedSparseNet[validInteractions,:]
        uniqueTFs = unique(intersect(gs[:,1],TFnames))
        uniqueGenes = unique(intersect(gsTargs,geneNames))
        validGSInteractions = intersect(find(indexin(gs[:,1],TFnames)),
            find(indexin(gs[:,2], geneNames)))
        totalGSInteractions = length(validGSInteractions)
        validGS = gs[validGSInteractions,:]
        nInferred = size(validNet,1)
        if makePlot
            println("MTL inferred: ", nInferred, " potential interactions that overlap with gs")
        end
        infTuples = Array{Tuple{String,String},1}(nInferred)
        for i = 1:nInferred
            infTuples[i] = (validNet[i,[1,2]]...)
        end
        TEdges = size(validGS, 1)
        gsTuples = Array{Tuple{String,String},1}(TEdges)
        for i = 1:TEdges
            gsTuples[i] = (validGS[i,:]...)
        end
        InGs = function(x)
            in(x, gsTuples)
        end
        InGsVec = map(InGs, infTuples)
        InInf = function(x)
            in(x, infTuples)
        end
        InInfVec = map(InInf, gsTuples)
        if k == 1
            InInfMat = Array{Any, 2}(length(InInfVec),ntasks+1)
            InInfMat[:,1] = gsTuples
        end
        InInfMat[:,k+1] = InInfVec
        TP = sum(InGsVec)
        FP = nInferred - TP
        FN = TEdges - TP
        TotalNegs = length(uniqueTFs)*length(uniqueGenes) - TEdges
        TN = TotalNegs - FP
        if makePlot
            println(TP, " of these edges were correctly inferred")
        end
        confLevels = unique(validNet[:,rankCol])
        totLevels = length(confLevels)
        precisions = zeros(totLevels)
        recalls = zeros(totLevels)
        for lev = 1:totLevels
            levelInds = find(validNet[:,rankCol] .>= confLevels[lev])
            levTP = sum(InGsVec[levelInds])
            precisions[lev] = levTP/length(levelInds)
            recalls[lev] = levTP/TEdges
        end
        if extrapolation == true
            precisions = [precisions[1]; precisions;TEdges/(length(uniqueTFs)*length(uniqueGenes))]
            recalls = [0; recalls;1]
        else
            if length(precisions) > 0
                precisions = [precisions[1]; precisions]
            else
                precisions = [0; precisions]
            end
            recalls = [0; recalls]
        end
        heights = (precisions[2:end] + precisions[1:(end - 1)])/2
        widths = recalls[2:end] - recalls[1:(end-1)]
        AUPRs[k] = heights' * widths
        MCC = ((TP*TN) - (FP*FN))/
            sqrt((TP + FP)*(TP+FN)*(TN+FP)*(TN+FN))
        if isnan(MCC)
            MCC = -1.
        end
        MCCs[k] = MCC
        F1s[k] = 2*TP/(2*TP+FN+FP)
        if makePlot
            if k == 1
                plot!([0,1], [TEdges/(length(uniqueTFs)*length(uniqueGenes)),
                    TEdges/(length(uniqueTFs)*length(uniqueGenes))],
                    label = "Random", linewidth = 3, linestyle = :dash, color = :darkgray)
            end
            plot!(p, recalls,precisions, xlims = [0,1], ylims = [0,1],
                label = TaskNames[k] * " (" * string(round(AUPRs[k], 4)) *")",
                xlabel = "Recall", ylabel = "Precision", linewidth = 3, linealpha = 0.8,
                color = colors[mod(k-1,ntasks)+1])
        end
    end
    if makePlot
        display(p)
    end
    if length(TaskNames) > 0
        InInfMat = [reshape(vcat("TF, Gene",TaskNames),(1,1+ntasks));InInfMat]
    end
    return AUPRs, MCCs, F1s, p, InInfMat
end

function getGScomparison(NetsOutputFiles::Array{String, 1},
    NetsOutputMat::String, gsFile::String, gsTargsFile::String, rankCol::Int64,
    extrapolation::Bool, GsOutputDir::String;inputNames::Array{String,1} = Array{String,1}(0))
    println("Loading networks")
    NetsMat = read_matfile(NetsOutputMat)
    if length(inputNames) == 0
        TaskNames = map(string, jvector(NetsMat["TaskNames"]))
    else
        TaskNames = inputNames
    end
    geneNames = map(string, jvector(NetsMat["geneNames"]))
    TFNames = map(string, jvector(NetsMat["TFNames"]))
    ntasks = length(TaskNames)
    sparseNets = [readdlm(NetPath)[2:end,:] for NetPath = NetsOutputFiles]
    gs = readdlm(gsFile,String)
    gs = gs[2:end,1:2]
    gs_Targs = readdlm(gsTargsFile, String)[:]
    AUPRs, MCCs, F1s, p, InInfMat = getComparison(sparseNets, gs, geneNames, TFNames, rankCol,
        gs_Targs, TaskNames = TaskNames, extrapolation = extrapolation, makePlot = true)
    savefig(p, GsOutputDir*"PR.pdf")
    write_matfile(GsOutputDir*"scores.mat";
            Tasks = TaskNames,
            AUPRs = AUPRs,
            MCCs = MCCs,
            F1s = F1s
    )
    writedlm(GsOutputDir*"InferredEdges.tsv",InInfMat)
    println("AUPR: ", round(mean(AUPRs), 3)," | MCC: ", round(mean(MCCs), 3),
        " | F1: ", round(mean(F1s), 3))
end
