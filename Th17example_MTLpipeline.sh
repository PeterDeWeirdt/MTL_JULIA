#!/usr/bin/env bash
matlab="/Applications/MATLAB_R2016a.app/bin/matlab"
julia="/Applications/Julia-0.6.app/Contents/Resources/julia/bin/julia"
echo "Multitask Learning Pipeline"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "Setting up data using MATLAB"
$matlab -nodisplay -nosplash -nodesktop -r "run('Th17example_setup.m');exit;"
echo "Setup file names output to setup.txt"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "Creating networks using multitask learning in Julia"
$julia "./Add_packages.jl" # Remove after first run
$julia "./Th17example_inference.jl"
