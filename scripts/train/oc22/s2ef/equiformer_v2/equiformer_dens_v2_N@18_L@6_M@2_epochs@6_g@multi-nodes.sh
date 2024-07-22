python main.py \
    --mode 'train' \
    --distributed \
    --num-gpus 8 \
    --num-nodes 4 \
    --config-yml 'experimental/configs/oc22/equiformer_v2/equiformer_v2_dens_N@18_L@6_M@2_e@4_f@100_std@0.15.yml' \
    --identifier 'oc22_equiformer-v2-dens' \
    --run-dir 'models/oc22/equiformer_v2_dens/' \
    --submit \
    --slurm-mem 480 \
    --amp