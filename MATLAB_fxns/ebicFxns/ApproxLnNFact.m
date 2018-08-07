function ln_n_fact = ApproxLnNFact(n)
%% Goal: Use the sterling approximation to get ln(n!)
    ln_n_fact = n*log(n) - n+ 0.5*log(2*pi*n);
end