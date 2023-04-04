2023年03月27日
./run_greaselm.sh obqa --data_dir data/ --emp True --use_wandb True 
./run_greaselm.sh obqa --data_dir data/ --emp False --use_wandb True 

2023年03月28日
./run_greaselm.sh csqa --data_dir data/ --emp True --use_wandb True 
./run_greaselm.sh csqa --data_dir data/ --emp False --use_wandb True 

看obqa的k的值的影响
./run_greaselm.sh obqa --data_dir data/ --emp True --use_wandb True -k 1
./run_greaselm.sh obqa --data_dir data/ --emp False --use_wandb True -k 1

./run_greaselm.sh obqa --data_dir data/ --emp True --use_wandb True -k 3
./run_greaselm.sh obqa --data_dir data/ --emp True --use_wandb True -k 7

./run_greaselm.sh obqa --data_dir data/ --emp False --use_wandb True -k 3
./run_greaselm.sh obqa --data_dir data/ --emp False --use_wandb True -k 7



2023年03月29日
./run_greaselm.sh obqa --data_dir data/ --emp False --use_wandb True --mix_number 2 
./run_greaselm.sh obqa --data_dir data/ --use_wandb True --all_mix True 

2023年04月03日
./run_greaselm.sh obqa --data_dir data/ --emp False --use_wandb True --mix_number 2
./run_greaselm.sh obqa --data_dir data/ --emp False --use_wandb True --mix_number 3
./run_greaselm.sh obqa --data_dir data/ --emp False --use_wandb True --mix_number 5
./run_greaselm.sh obqa --data_dir data/ --emp False --use_wandb True --mix_number 10 
./run_greaselm.sh obqa --data_dir data/ --emp False --use_wandb True --all_mix True (需要重新测试)

./run_greaselm.sh obqa --data_dir data/ --emp False --use_wandb True --gnn_dim 100
./run_greaselm.sh obqa --data_dir data/ --emp False --use_wandb True --gnn_dim 50
./run_greaselm.sh obqa --data_dir data/ --emp False --use_wandb True --gnn_dim 300
./run_greaselm.sh obqa --data_dir data/ --emp False --use_wandb True --gnn_dim 400


01245





