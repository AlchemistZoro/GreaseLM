export TOKENIZERS_PARALLELISM=true
dt=`date '+%Y%m%d_%H%M%S'`

dataset=$1
shift
args=$@
ebs=32

# Added for eval
mode=eval

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "******************************"

run_name=eval__conceptlm__${dataset}__${dt}

###### Eval ######
python3 -u conceptlm.py \
    --run_name ${run_name} \
    --mode ${mode} -ebs ${ebs} --dataset ${dataset} \
    $args
