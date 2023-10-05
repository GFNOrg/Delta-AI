python main_sampler.py \
    --exp_name MCMC_$1_ising_$2_vdim$3_sigma$4_run$5 \
    --alg $1 \
    --model ising \
    --graph $2 \
    --vdim $3 \
    --n_iters 300000 \
    --batchsz 1000 \
    --sigma $4 \
    --print_every 500
