# MTL_JULIA
MTL_JULIA is a pipeline for transcriptional regulatory network (TRN) inference using multitask learning (MTL) written in MATLAB and julia. 

## References 
1. Castro, Dayanne M., et al. "Multi-study inference of regulatory networks for more accurate models of gene regulation." bioRxiv (2018): 279224.
2. Miraldi, Emily R., et al. "Leveraging chromatin accessibility for transcriptional regulatory network inference in T Helper 17 Cells." bioRxiv (2018): 292987.

## Highlights
1. Written in Julia for speed. For reference, TRN inference on the Th17 dataset took 37 minutes on a macbook pro running on 6 core processors. 
2. Parallel implementation 
3. Parameter selection with EBIC or cross validation

![](/images/MTL_TRN_Inference_Workflow.png)

## Installation 
1. You need a licensed version of MATLAB
2. Download [JuliaPro](https://juliacomputing.com/products/juliapro.html) Version 0.6.3 or greater (there are [other](https://julialang.org/downloads/) download options too)
3. Download this github repository
3. From julia run "Add_packages.jl" to install julia packages

## Th17 Example
#### Interactive (recomended for first run)
First we use MATLAB for Transcription factor estimation and prior matrix creation. 
1. Open "Th17example_setup.m"
2. Set options in the script - if you would like to run MATLAB serially:
```matlab
parallel = false;
```
3. Run "Th17example_setup.m" 
Now we use the outputs from MATLAB for network inference in Julia. 
*Note: Julia reads the filepaths for the MATLAB outputs from "setup.txt" in the setup folder, so no user specification is necessary*
4. Open "Th17example_inference.jl"
5. Set options in the script - if you would like to run Julia serially: 
```julia
parallel = false
```
or with a different number of processors:
```julia
Nprocs = 2 
```
If you would like to check the number of processors on your machine, in Julia you can type
```julia
Sys.CPU_CORES 
```
There are two main parameter selection strategies to choose from:
##### Extended Bayesian Information Criteria
```julia
Fit = :ebic
getFitsParallel(DataMatPaths, Fit, Smin, Smax, Ssteps, nB, TaskNames, FitsOutputDir,
        FitsOutputMat, tolerance = tolerance, useBlockPrior = useBlockPrior)
```
##### Cross Validation
```julia
Fit = :cv
nfolds = 2
getFitsParallel(DataMatPaths, Fit, Smin, Smax, Ssteps, nB, TaskNames, FitsOutputDir,
        FitsOutputMat, tolerance = tolerance, useBlockPrior = useBlockPrior, nfolds = nfolds)
```
6. Run "Th17example_inference.jl"
7. Check the outputs folder for outputs

#### Shell Script
The above steps are autonomized in the shell script "Th17example_MTLpipeline.sh." Before running, you will likely have to set the matlab and julia binary paths:
```shell
matlab="/Applications/path/to/bin/matlab"
julia="/Applications/path/to/bin/julia"
```

