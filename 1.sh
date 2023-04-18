# CUDA_VISIBLE_DEVICES=5 ./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True -k 1
# CUDA_VISIBLE_DEVICES=5 ./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True -k 3
# CUDA_VISIBLE_DEVICES=5 ./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True -k 7
# CUDA_VISIBLE_DEVICES=5 ./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True --mix_number 2
# CUDA_VISIBLE_DEVICES=5 ./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True --mix_number 3
CUDA_VISIBLE_DEVICES=5 ./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True --mix_number 5
# CUDA_VISIBLE_DEVICES=6 ./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True --mix_number 10 
# CUDA_VISIBLE_DEVICES=6 ./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True --mix_number 20