# MTL_JULIA
MTL_JULIA is a pipeline for transcriptional regulatory network (TRN) inference using multitask learning (MTL) written in MATLAB and julia. 

## References 
1. Castro, Dayanne M., et al. "Multi-study inference of regulatory networks for more accurate models of gene regulation." bioRxiv (2018): 279224.
2. Miraldi, Emily R., et al. "Leveraging chromatin accessibility for transcriptional regulatory network inference in T Helper 17 Cells." bioRxiv (2018): 292987.

## Highlights
1. Written in Julia for speed
2. Parallel and serial implementation options
3. Variety of parameter options for inference (see **Parameter Options**)

![](/images/MTL_TRN_Inference_Workflow.png)

## Installation 

## Basic Usage

## Parameter Options 
### MATLAB
1. TFA estimation or TF_mRNA: "TFA" or TF_mRNA
```MATLAB 
TFA = ''
``` 
### JULIA
```julia
fitMethod = :ebic
```
