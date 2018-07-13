#Peter DeWeirdt
#07/13/2018

#Pkg.add("Plots")
#Pkg.add("StatPlots")
#Pkg.add("MAT")
#Pkg.add("OhMyREPL")
using OhMyREPL
using StatsBase
using Plots
using ProgressMeter
using StatPlots
using MAT
using IterTools
pyplot()

function preprocess_data(Xs::Array{Array{Float64,2},1}, Ys::Array{Array{Float64,1},1};
    score_X = true, score_Y = true)
    """z-score data"""
    ntasks = length(Xs)
    for k in 1:ntasks
        X = Xs[k]
        Y = Ys[k]
        if score_X
            Xs[k] = zscore(X, mean(X,1), std(X, 1, corrected = false))
        end
        if score_Y
            Ys[k] = zscore(Y, mean(Y,1), std(Y,1, corrected = false))
        end
    end
    return Xs, Ys
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
    # Devctorized for speed
    for k in 1:ntasks
        X = Xs[k]
        Y = Ys[k]
        if calcCs;Cs[k] = X'*Y;end
        if calcDs;Ds[k] = X'*X;end
    end
    return Cs, Ds
end

function updateS(Cs::Array{Array{Float64,1},1}, Ds::Array{Array{Float64,2},1},
    B::Array{Float64,2}, S::Array{Float64,2}, lamS::Float64, P::Array{Float64,2})
    """returns updated coefficients for S-- (sparse matrix: predictors x tasks)
    lasso regularized -- using cyclical coordinate descent and
    soft-thresholding"""
    ntasks = length(Cs)
    npreds = size(P, 1)
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
    tolerance = 1e-7, score = false, ntasks = nothing, npreds = nothing)
    """Fits regression model in which the weights matrix W (predictors x tasks)
    is decomposed in two components: B that captures block structure across tasks
    and S that allows for the differences.
    reference: Jalali et al., NIPS 2010. A Dirty Model for Multi-task Learning."""
    if ntasks == nothing;ntasks = length(Xs);end
    if npreds == nothing;npreds = size(Xs[1], 2);end
    if score
        Xs, Ys = preprocess_data(Xs, Ys)
    end
    if P == nothing; P = ones(npreds, ntasks);end
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
            println(maximum(abs.(W-W_old)))
            println("Maxed")
            println(tolerance)
            println(W)
            println(W_old)
        end
    end
    SOut[original_indices,:] = S
    BOut[original_indices,:] = B
    return(W, BOut, SOut)
 end


function test_MTL(ntasks::Int64, npreds::Int64, nsamps::Array{Int64, 1},
    n_nonzero::Int64; lamB = 0., lamS = 0., tolerance = nothing, maxiter = 1000)
    Xs = Array{Array{Float64,2},1}(ntasks)
    Ys = Array{Array{Float64,1},1}(ntasks)
    Ws = zeros(npreds, ntasks)
    srand(1)
    for i = 1:ntasks
        X = rand(nsamps[i],npreds)
        W = zeros(npreds)
        W[sample(1:npreds, n_nonzero, replace = false)] = 1
        Ws[:,i] = W
        Y = X*W
        Xs[i] = X;Ys[i] = Y
    end
    t = @elapsed W_inf,S,B = dirty_multitask_lasso(Xs, Ys, lamB = lamB, lamS = lamS,
    tolerance = tolerance, maxiter = maxiter)
    println(sum(Ws - abs.(sign.(W_inf))))
    return(t)
end

function get_times(tasks, preds, samps, d_task, d_pred, d_samp;
    nzero = 15, lamB = 0.02, lamS = 0.01, maxiter = 1000, tolerance = nothing)
    task_time = zeros(length(tasks))
    for i = 1:length(tasks)
        ntask = tasks[i]
        d_samps = repmat([d_samp], ntask)
        task_time[i] = test_MTL(ntask, d_pred, d_samps, nzero, lamB = lamB, lamS = lamS, maxiter = maxiter, tolerance = tolerance)
    end
    plot(tasks,task_time, marker = (2,:circle), line = :dot, xlabel = "Tasks", ylabel = "Time (seconds)")
    png("tasks")

    d_samps = repmat([d_samp], d_task)
    pred_time = zeros(length(preds))
    for i = 1:length(preds)
        pred = preds[i]
        pred_time[i] = test_MTL(d_task, pred, d_samps, nzero, lamB = lamB, lamS = lamS, tolerance = tolerance)
    end
    plot(preds,pred_time, m = (2,:circle), line = :dot, xlabel = "Predictors", ylabel = "Time (seconds)")
    png("preds")

    samp_time = zeros(length(samps))
    for i = 1:length(samps)
        samp = repmat([samps[i]], d_task)
        samp_time[i] = test_MTL(d_task, d_pred, samp, nzero, lamB = lamB, lamS = lamS, tolerance = tolerance)
    end
    plot(samps,samp_time, m = (2, :circle), line = :dot, xlabel = "Samples", ylabel = "Time (seconds)")
    png("samps")
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
        tot_preds = length(W[:,k])
        nonzero_pred = sum(Int,abs.(sign.(W[:,k])))
        RSS = get_RSS(Xs[k],Ys[k],W[:,k])
        BIC_penalty = nonzero_pred*log(samps)
        if nonzero_pred == 0 || nonzero_pred == tot_preds
            EBIC_penalty = 0
        else
            EBIC_penalty = 2*gamma*(approxLnNFact(tot_preds) -
            (approxLnNFact(nonzero_pred) + approxLnNFact((tot_preds - nonzero_pred))))
        end
        # println("\tTask: ", k, " Nonzero: ", sum(abs.(sign.(W))), "\n\tRSS: ", RSS, " RSS_pen: ", samps*log(RSS/samps),
        # "\n\tBIC: ", BIC_penalty, " EBIC: ", EBIC_penalty)
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

function fit_gene(Xs::Array{Array{Float64,2},1}, Ys::Array{Array{Float64,1},1},
    Ds::Array{Array{Float64,2},1},ntasks::Int64, npreds::Int64,
    nsamps::Array{Int64,1}, lamSs::Array{Float64,1};nB = 3,
    prior = nothing, fit = :ebic, nfolds = 5)
    """For one gene, calculate fit for each pair of lamS, lamB
    fit options -- :ebic and :cv
    Note: sliding window for lambdaB"""
    if fit == :ebic
        Cs,~ = covariance_update_terms(Xs, Ys, calcDs = false, calcCs = true)
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
            if fit == :ebic
                W,B,S = dirty_multitask_lasso(Xs, Ys;
                    P = prior, lamB = lamB, lamS = lamS,
                    Cs = Cs, Ds = Ds, S = S, B = B, ntasks = ntasks, npreds = npreds)
                currFit = ebic(Xs, Ys, W, ntasks, nsamps, npreds)
            elseif fit == :cv
                folds = Array{Array{Array{Int64,1},1},1}(ntasks)
                for task = 1:ntasks
                    folds[task] = kfoldperm(nsamps[task], nfolds)
                end
                foldErrors = Array{Float64, 1}(nfolds)
                for fold = 1:nfolds
                    LOXs = Array{Array{Float64,2},1}(ntasks)
                    LOYs = Array{Array{Float64,1},1}(ntasks)
                    InXs = Array{Array{Float64,2},1}(ntasks)
                    InYs = Array{Array{Float64,1},1}(ntasks)
                    for task = 1:ntasks
                        currFold = folds[task][fold]
                        LOXs[task] = Xs[task][currFold,:]
                        LOYs[task] = Ys[task][currFold]
                        keepInds = (1:nsamps[task])[filter((x) -> !(x in currFold),1:nsamps[task])]
                        InXs[task] = Xs[task][keepInds,:]
                        InYs[task] = Ys[task][keepInds]
                    end
                    W,B,S = dirty_multitask_lasso(InXs, InYs;
                        P = prior, lamB = lamB, lamS = lamS,
                        S = S, B = B, ntasks = ntasks, npreds = npreds)
                    sq_err = 0
                    for k = 1:ntasks
                        sq_err += get_RSS(LOXs[k],LOYs[k],W[:,k])
                        if sq_err < 0
                            println(get_RSS(LOXs[k],LOYs[k],W[:,k]))
                        end
                    end
                    foldErrors[fold] = sq_err
                end
                currFit = mean(foldErrors)
            else
                println("Fit method not supported")
            end
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
    YSs::Array{Array{Array{Float64,1},1},1}; Smin = 0.01, Smax = 1,
    Ssteps = 10, nB = 4, prior = nothing, fit = :ebic)
    """Calculate fit for each gene in a network. Plot grid of median fits.
    Return lambdaS and lambdaB within 1 se of the minimum fit."""
    ntasks = length(Xs)
    Xs,~ = preprocess_data(Xs, YSs[1], score_X = true, score_Y = false)
    ~,Ds = covariance_update_terms(Xs, YSs[1], calcCs = false, calcDs = true)
    npreds = size(Xs[1], 2)
    nsamps = Array{Int64, 1}(ntasks)
    for task = 1:ntasks
        nsamps[task] = size(Xs[task], 1)
    end
    ngenes = length(YSs)
    lamSlog10step = 1/Ssteps
    logLamSRange = log10(Smin):lamSlog10step:log10(Smax)
    lamSs = 10.^logLamSRange
    geneFits = Array{Float64,2}(length(lamSs)*nB,ngenes)
    lambdas = Array{Float64,2}(length(lamSs)*nB,2)
    println("Estimating fits for " * string(ngenes) * "genes")
    @showprogress 1 for genei = 1:ngenes
        ~,Ys = preprocess_data(Xs, YSs[genei], score_X = false, score_Y = true)
        Fits, lambdas = fit_gene(Xs, Ys, Ds,ntasks, npreds, nsamps, lamSs,
        nB = nB, prior = prior, fit = fit)
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
    p::Plots.Plot = plot(lambdas[:,1],lambdas[:,2], seriestype=:scatter,marker_z = networkFits,
    xlabel = "lambda S", ylabel = "lambda B", label = "", colorbar_title = "Fit",
    xscale = :log10, yscale = :log10, xlims = (10^(log10(minimum(lambdas[:,1]))-0.1),
    10^(log10(maximum(lambdas[:,1]))+0.1)),
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

function simulate_data(ntasks::Int64, npreds::Int64, nsamps::Array{Int64, 1},
    n_nonzero::Int64, ngenes::Int64)
    Xs = Array{Array{Float64,2},1}(ntasks)
    YSs = Array{Array{Array{Float64,1},1}}(ngenes)
    networks = Array{Array{Float64,2},1}(ntasks)
    for i = 1:ntasks
        X = rand(nsamps[i],npreds)
        Xs[i] = X;
        networks[i] = zeros(ngenes, npreds)
    end
    for gene = 1:ngenes
        Ys = Array{Array{Float64,1},1}(ntasks)
        Ws = zeros(npreds, ntasks)
        coreWs = sample(1:npreds, n_nonzero, replace = false)
        for task = 1:ntasks
            BWs = sample(coreWs, floor(Int64, n_nonzero/2), replace = false)
            SWs = sample(1:npreds, n_nonzero - floor(Int64, n_nonzero/2), replace = false)
            W = zeros(npreds)
            W[union(BWs, SWs)] = 1
            networks[task][gene,:] = W
            Ws[:,task] = W
            Y = Xs[task]*W + 0.1*rand(nsamps)
            Ys[task] = Y
        end
        YSs[gene] = Ys
    end
    return(Xs, YSs, networks)
end

function GetBestNets(Xs::Array{Array{Float64,2},1}, YSs::Array{Array{Array{Float64,1},1}},
    lamS::Float64, lamB::Float64; prior = nothing)
    ntasks = length(Xs)
    Xs,~ = preprocess_data(Xs, YSs[1], score_X = true, score_Y = false)
    ~,Ds = covariance_update_terms(Xs, YSs[1], calcCs = false, calcDs = true)
    npreds = size(Xs[1], 2)
    ngenes = length(YSs)
    networks = Array{Array{Float64,2},1}(ntasks)
    for task = 1:ntasks
        networks[task] = zeros(ngenes, npreds)
    end
    for genei = 1:ngenes
        ~,Ys = preprocess_data(Xs, YSs[genei], score_X = false, score_Y = true)
        Cs,~ = covariance_update_terms(Xs, Ys, calcDs = false, calcCs = true)
        W,B,S = dirty_multitask_lasso(Xs, Ys;
            P = prior, lamB = lamB, lamS = lamS,
            Cs = Cs, Ds = Ds, ntasks = ntasks, npreds = npreds)
        for task = 1:ntasks
            networks[task][genei,:] = W[:,task]
        end
    end
    return networks
end

function buildTRNs(Xs::Array{Array{Float64,2},1},
    YSs::Array{Array{Array{Float64,1},1},1}, lamS, lamB; nboots = 50)
    """Rank TF-gene interactions according to confidence:
    1 - var(residuals_i)/var(residual[~TF])."""
end



function test_fit()
    Xs,YSs,true_nets =  simulate_data(2, 50, [50,50],15,2)
    lambdas = fit_network(Xs, YSs, fit = :ebic)
    optLamS = lambdas[1]
    optLamB = lambdas[2]
    optimal_nets = GetBestNets(Xs, YSs, optLamS, optLamB)
    hamming_dist = Array{Int64,1}(length(optimal_nets))
    for i = 1:length(optimal_nets)
        println(optimal_nets[i])
        println(true_nets[i])
        hamming_dist[i] = sum(abs.(abs.(sign.(true_nets[i])) - abs.(sign.(optimal_nets[i]))))
    end
    return hamming_dist
end

function time_one_gene(ntrials)
    ts = Array{Float64, 1}(ntrials)
    hammings = Array{Float64, 1}(ntrials)
    @showprogress 1 for i = 1:ntrials
        Xs,YSs,true_nets =  simulate_data(2, 300, [5000,250],15,1)
        ts[i] = @elapsed optimal_nets = GetBestNets(Xs, YSs, 0.01, 0.02)
        hamming_dist = Array{Int64,1}(length(optimal_nets))
        for j = 1:length(optimal_nets)
            hamming_dist[j] = sum(abs.(abs.(sign.(true_nets[j])) - abs.(sign.(optimal_nets[j]))))
        end
        hammings[i] = mean(hamming_dist)
    end
    tPlot = boxplot(ts, label = "", ylabel = "time (seconds)", markeralpha = 0.6)
    hPlot = boxplot(hammings, label = "", ylabel = "hamming dist.", markeralpha = 0.6)
    png(tPlot, "oneGeneSimTimes")
    png(hPlot, "oneGeneSimHamming")
    display(tPlot)
    display(hPlot)
    return ts, hammings
end

function get_partitions(data::Array{Any,1}, nparts::Int64)
    #Assume length(data) %% nparts = 0
    partitionedData = Array{Array{Any,1},1}(nparts)
    dataLength = length(data)
    partLength = dataLength/nparts
    for part = 1:nparts
        partitionedData[part] = data[((part-1)*partLength + 1):part*partLength]
    end
    return(partitionedData)
end

function MTL()
    """MAIN: fit TRNs for multiple tasks using multitask learning"""
    TaskMatPaths = ["./fitSetup/RNAseq_ATAC_Th17_bias50.mat",
        "./fitSetup/scRNAseq_ATAC_Th17_bias50.mat"]
    ntasks = length(TaskMatPaths)
    nsamps = Array{Int64}(ntasks)
    ngenes = Int64
    nTFs = Int64
    taskMTLinputs = Array{Dict{String,Any},1}(ntasks)
    Xs = Array{Array{Float64,2},1}(ntasks)
    tempYSs = Array{Float64,2}(0,0)
    prior = Array{Float64,2}(0,0)
    @showprogress for task = 1:ntasks
        inputs = matread(TaskMatPaths[task])
        taskMTLinputs[task] = inputs
        Xs[task] = inputs["predictorMat"]'
        currSamps = size(Xs[task],1)
        nsamps[task] = currSamps
        if task == 1
            prior = inputs["priorWeightsMat"]'
            ngenes = size(prior,2)
            nTFs = size(prior,1)
            tempYSs = inputs["responseMat"]'
        end
        [tempYSs;inputs["responseMat"]']
    end

    println("Getting gene responses")
    YSs = Array{Array{Array{Float64,1},1}}(ngenes)
    @showprogress for gene = 1:ngenes
        YSs[gene] = get_partitions(tempYSs[:,gene])
    end
    lams =  fit_network(Xs, YSs::Array{Array{Array{Float64,1},1},1}, Smin = 0.01,
    Smax = 1, Ssteps = 10, nB = 4, prior = prior, fit = :ebic)
end
