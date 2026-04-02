dataset="ZDXX"                    # ZD_SCFC, HCP, ZD, XX, ZDXX save path
dataset_name="merge_dataset"          # load in dataset path []
device=0
date=`date +%F`
root=`pwd`
hidden_dim=128
fusion_hidden=128
num_fusion_layers=2
batch_size=128
num_experts=2
k=2
lr=0.0005
dropout=0.5
alpha=0.3
mode="KD"                    # decide KD or not [KD, train]
fusion="MoE"                    # [MoE, MLP]
stage="moe_pretrain"                     # decide use which loss function [moe_pretrain, mix]
backbone="GCN"
epochs=500
patience=300
id_proc_per_card=2

nohup python -u main_mmoe.py\
    --dataset ${dataset} \
    --device ${device} \
    --root ${root} \
    --path ${root}/data/${dataset_name} \
    --result_path ${root}/logs/results/${dataset}_${device}_${id_proc_per_card}.txt \
    --num_fusion_layers ${num_fusion_layers} \
    --batch_size ${batch_size} \
    --hidden_dim ${hidden_dim} \
    --fusion_hidden ${fusion_hidden} \
    --k ${k} \
    --lr ${lr} \
    --dropout ${dropout} \
    --alpha ${alpha} \
    --mode ${mode} \
    --fusion ${fusion} \
    --stage ${stage} \
    --backbone ${backbone} \
    --epochs ${epochs} \
    --patience ${patience} \
    > ${root}/logs/out/${date}_${dataset}_${device}_${id_proc_per_card}.log  2>&1 &