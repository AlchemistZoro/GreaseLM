# ConceptLM

## 1. Dependencies

- [Python](<https://www.python.org/>) == 3.8
- [PyTorch](<https://pytorch.org/get-started/locally/>) == 1.8.0
- [transformers](<https://github.com/huggingface/transformers/tree/v3.4.0>) == 3.4.0
- [torch-geometric](https://pytorch-geometric.readthedocs.io/) == 1.7.0

Run the following commands to create a conda environment (assuming CUDA 10.1):
```bash
conda create -y -n conceptlm python=3.8
conda activate conceptlm
pip install numpy==1.18.3 tqdm
pip install torch==1.8.0+cu101 torchvision -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==3.4.0 nltk spacy
pip install wandb
conda install -y -c conda-forge tensorboardx
conda install -y -c conda-forge tensorboard

# for torch-geometric
pip install torch-scatter==2.0.7 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install torch-geometric==1.7.0 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
```


## 2. Download data

### Download and preprocess data yourself
**Preprocessing the data yourself may take long, so if you want to directly download preprocessed data, please jump to the next subsection.**

Download the raw ConceptNet, CommonsenseQA, OpenBookQA data by using
```
./download_raw_data.sh
```

You can preprocess these raw data by running
```
CUDA_VISIBLE_DEVICES=0 python preprocess.py -p <num_processes>
```
You can specify the GPU you want to use in the beginning of the command `CUDA_VISIBLE_DEVICES=...`. The script will:
* Setup ConceptNet (e.g., extract English relations from ConceptNet, merge the original 42 relation types into 17 types)
* Convert the QA datasets into .jsonl files (e.g., stored in `data/csqa/statement/`)
* Identify all mentioned concepts in the questions and answers
* Extract subgraphs for each q-a pair

The script to download and preprocess the [MedQA-USMLE](https://github.com/jind11/MedQA) data and the biomedical knowledge graph based on Disease Database and DrugBank is provided in `utils_biomed/`.

### Directly download preprocessed data
For your convenience, if you don't want to preprocess the data yourself, you can download all the preprocessed data [here](https://drive.google.com/drive/folders/1T6B4nou5P3u-6jr0z6e3IkitO8fNVM6f?usp=sharing). Download them into the top-level directory of this repo and unzip them. Move the `medqa_usmle` and `ddb` folders into the `data/` directory.



## 3. Training ConceptLM
To train ConceptLM on CommonsenseQA, run
```
CUDA_VISIBLE_DEVICES=0 ./run_conceptlm.sh csqa --data_dir data/
```

Debug on OBQA
```
CUDA_VISIBLE_DEVICES=0 ./run_conceptlm.sh obqa --data_dir data/  --emp True --debug True
```


## 4. Experimental expansion
### BASE MODEL
```
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True 
./run_conceptlm.sh csqa --data_dir data/ --emp False --use_wandb True 
```

### Different number of mixed coding layers
```
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True -k 1
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True -k 3
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True -k 7
```
### Entity encoding node
```
./run_conceptlm.sh obqa --data_dir data/ --emp True --use_wandb True -k 1
./run_conceptlm.sh obqa --data_dir data/ --emp True --use_wandb True -k 3
./run_conceptlm.sh obqa --data_dir data/ --emp True --use_wandb True -k 5
./run_conceptlm.sh obqa --data_dir data/ --emp True --use_wandb True -k 7
./run_conceptlm.sh csqa --data_dir data/ --emp True --use_wandb True -k 5
```
### Different number of interaction nodes
```
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True --mix_number 2
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True --mix_number 3
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True --mix_number 5
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True --mix_number 10 
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True --mix_number 20
```
### Subgraphs of different number of nodes
```
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True --gnn_dim 100
./run_conceptlm.sh obqa --data_dir data/ --emp False --use_wandb True --gnn_dim 300
```

