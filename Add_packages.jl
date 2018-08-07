Packages = ["StatsBase", "Plots", "ProgressMeter", "StatPlots", "MAT",
    "IterTools", "GLM"]
println("Adding "*length(Packages)* " packages")
for package = Packages
    println("package: "*package)
    try Pkg.installed(package)
    catch Pkg.add(package)
    end
end
println("Done adding packages")
