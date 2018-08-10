# Add_packages.jl
# Packages for MTL TRN Inference
# Author: Peter DeWeirdt

Packages = ["StatsBase", "Plots", "ProgressMeter", "StatPlots", "MATLAB",
    "IterTools", "GLM"]
println("Adding "*string(length(Packages))* " packages")
for i = 1:length(Packages)
    println("Adding package "*string(i)*" of "*string(length(Packages))*" ~~ "*Packages[i]*" ~~ ")
    Pkg.add(Packages[i])
end

println("Done adding packages")
