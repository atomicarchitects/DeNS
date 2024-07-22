python main.py \
    --mode 'train' \
    --distributed \
    --num-gpus 8 \
    --num-nodes 4 \
    --config-yml 'experimental/configs/oc20/2M/equiformer_v2/equiformer_dens_v2_N@12_L@6_M@2_lr@4e-4_epochs@30_std@0.15_gpus@32.yml' \
    --identifier 'oc20-2M_equiformer-v2-dens_epochs@30' \
    --run-dir 'models/oc20/2M/equiformer_v2_dens/epochs@30/' \
    --submit \
    --slurm-mem 480 \
    --amp