"""
compare_fits.jl
"""

MCC_fit = -matread("EstimatedFits_MCC.mat")["networkFits"]
BIC_fit = matread("EstimatedFits_bic.mat")["networkFits"]
EBIC_fit = matread("EstimatedFits_ebic.mat")["networkFits"]
CV_fit = matread("EstimatedFits_cv.mat")["networkFits"]

scored_MCC = zscore(MCC_fit)
scored_BIC = zscore(BIC_fit)
scored_EBIC = zscore(EBIC_fit)
scored_CV = zscore(CV_fit)

plot(scored_MCC, scored_EBIC,  seriestype=:scatter, markeralpha = 0.5,
    markerstrokewidth = 0, markersize = 6, xlabel = "MCC", ylabel = "Estimate", label = "EBIC",
    markercolor = :skyblue, xlim = [-3,3], ylim = [-3,3])
EBIC_lm = lm(scored_EBIC, scored_MCC, false)
x = [minimum(scored_MCC), maximum(scored_MCC)]
plot!(x,x*coef(EBIC_lm)', linecolor = :skyblue, label = "", linewidth = 2)
#plot!(scored_MCC, scored_BIC,  seriestype=:scatter, markeralpha = 0.5,
#    markerstrokewidth = 0, markersize = 6, label = "BIC")
plot!(scored_MCC, scored_CV,  seriestype=:scatter, markeralpha = 0.5,
        markerstrokewidth = 0, markersize = 6, label = "2 fold CV", color = :indianred)
CV_lm = lm(scored_CV, scored_MCC, false)
plot!(x,x*coef(CV_lm)', linecolor = :indianred, label = "", linewidth = 2)
savefig("MCC_correlations.pdf")
