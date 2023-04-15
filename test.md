# 基础模型
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True 
./run_conceptlm.sh csqa --data_dir data/ --emp False --use_wandb True 

# 混合编码层数的影响
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True -k 1
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True -k 3
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True -k 7

# 实体编码节点的影响
./run_conceptlm.sh obqa --data_dir data/ --emp True --use_wandb True -k 1
./run_conceptlm.sh obqa --data_dir data/ --emp True --use_wandb True -k 3
./run_conceptlm.sh obqa --data_dir data/ --emp True --use_wandb True -k 5
./run_conceptlm.sh obqa --data_dir data/ --emp True --use_wandb True -k 7
./run_conceptlm.sh csqa --data_dir data/ --emp True --use_wandb True -k 5

## 不同节点数的实验
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True --mix_number 2
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True --mix_number 3
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True --mix_number 5
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True --mix_number 10 
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True --mix_number 20

## 不同子图个数的实验
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True --gnn_dim 100
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True --gnn_dim 300




