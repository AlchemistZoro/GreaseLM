CUDA_VISIBLE_DEVICES=7 ./run_conceptlm.sh obqa --data_dir data/ --emp True --use_wandb True -k 1
CUDA_VISIBLE_DEVICES=7 ./run_conceptlm.sh obqa --data_dir data/ --emp True --use_wandb True -k 3
CUDA_VISIBLE_DEVICES=7 ./run_conceptlm.sh obqa --data_dir data/ --emp True --use_wandb True -k 5
CUDA_VISIBLE_DEVICES=7 ./run_conceptlm.sh obqa --data_dir data/ --emp True --use_wandb True -k 7
CUDA_VISIBLE_DEVICES=7 ./run_conceptlm.sh csqa --data_dir data/ --emp True --use_wandb True -k 5