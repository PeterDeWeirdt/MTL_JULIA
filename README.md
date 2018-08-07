# MTL_JULIA
MTL_JULIA is a pipeline for transcriptional regulatory network (TRN) inference using multitask learning (MTL) written in MATLAB and julia. 

## References 
1. Castro, Dayanne M., et al. "Multi-study inference of regulatory networks for more accurate models of gene regulation." bioRxiv (2018): 279224.
2. Miraldi, Emily R., et al. "Leveraging chromatin accessibility for transcriptional regulatory network inference in T Helper 17 Cells." bioRxiv (2018): 292987.

## Highlights
1. Written in Julia for speed
2. Parallel and serial implementation options
3. Variety of parameter options for inference (see **Functionality**)

![](/images/MTL_TRN_Inference_Workflow.png)

## Installation 
1. You need a licensed version of MATLAB
2. Download [JuliaPro](https://juliacomputing.com/products/juliapro.html) Version 0.6.3 or greater (there are [other](https://julialang.org/downloads/) download options too).
3. Download this github repository.
3. From julia run *Add_packages.jl* to install julia packages.  

## Th17 Example
#### Interactive (recomended for first run)
First we use MATLAB for Transcription factor estimation and prior matrix creation. 
1. Open *Th17example_setup.m* 
2. Set options in the script - if you would like to run MATLAB serially:
```matlab
parallel = false;
```
3. Run *Th17example_setup.m* 
Now we use the outputs from MATLAB for network inference in Julia. 
*Note: Julia reads the filepaths for the MATLAB outputs from "setup.txt" in the setup folder, so we don't need to specify these.*
4. Open *Th17example_inference.jl*
5. Set options in the script - if you would like to run Julia serially: 
```julia
parallel = false
```
or with a different number of processors:
```julia
Nprocs = 2 
```
There are a depth of options for TRN inference using MTL, see **Functionality** for a description. 
6. Run *Th17example_inference.jl* 
7. Check the outputs folder for outputs

#### Shell Script
The above steps are autonomized in the shell script *Th17example_MTLpipeline.sh*. Before running, change the matlab and julia binary paths:
```shell
matlab="/Applications/path/to/bin/matlab"
julia="/Applications/path/to/bin/julia"
```

## Functionality
### Global
```julia
getFits = true
getNetworks = true
compareGS = true
``` 
### Parameter Selection
```julia
fitMethod = :ebic
```

### Network Inference

### Gold Standard Comparison
