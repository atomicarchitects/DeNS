python main.py \
    --mode 'train' \
    --distributed \
    --num-gpus 8 \
    --num-nodes 16 \
    --config-yml 'experimental/configs/oc20/all-md/equiformer_v2/equiformer_v2_dens_N@20_L@6_M@3_lr@4e-4_epochs@2_std@0.1_dens-relax-data-only.yml' \
    --identifier 'oc20-all-md_equiformer-v2-dens' \
    --run-dir 'models/oc20/all-md/equiformer_v2_dens/' \
    --submit \
    --slurm-mem 480 \
    --amp