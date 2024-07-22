python -u -m torch.distributed.launch --nproc_per_node=8 main.py \
    --mode 'train' \
    --distributed \
    --num-gpus 8 \
    --config-yml 'experimental/configs/oc20/2M/equiformer_v2/equiformer_dens_v2_N@12_L@6_M@2_lr@2e-4_epochs@12_std@0.1_gpus@16.yml' \
    --identifier 'oc20-2M_equiformer-v2-dens_epochs@12' \
    --run-dir 'models/oc20/2M/equiformer_v2_dens/epochs@12/' \
    --amp