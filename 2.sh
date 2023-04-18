CUDA_VISIBLE_DEVICES=5 ./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True 
CUDA_VISIBLE_DEVICES=5 ./run_conceptlm.sh csqa --data_dir data/ --emp False --use_wandb True 
CUDA_VISIBLE_DEVICES=6 ./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True --gnn_dim 100
CUDA_VISIBLE_DEVICES=6 ./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True --gnn_dim 300
