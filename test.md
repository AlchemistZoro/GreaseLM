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







