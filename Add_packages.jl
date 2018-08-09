Packages = ["StatsBase", "Plots", "ProgressMeter", "StatPlots", "MATLAB",
    "IterTools", "GLM"]
println("Adding "*string(length(Packages))* " packages")
for package = Packages
    println("package: "*package)
    Pkg.add(package)
    end
end
println("Done adding packages")
