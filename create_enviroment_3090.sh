# create enciroment for RTX3090
# CUDA version >= 11.1, torch version >=1.7.0
conda create -y -n glm python=3.8
conda activate glm

# should use conda to install pytorch, use pip will get the OSError
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch


# install torch-geometric from officiall doc
# https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

pip install scipy transformers==3.4.0 tensorboardx nltk spacy networkx wandb



