set -e

GPUS=0,1,2,3
NB_COMMA=`echo ${GPUS} | tr -cd , | wc -c`
NB_GPUS=$((${NB_COMMA} + 1))
PORT=$((9000 + RANDOM % 1000))

#shift

echo "Launching exp on $GPUS..."
CUDA_VISIBLE_DEVICES=${GPUS} python -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPUS} main.py \
--options options/data/imagenet100_50-10.yaml options/data/imagenet100_order1.yaml options/model/imagenet_dne.yaml \
    --name dne_imgnet100_mlp_dense_only_b50_10 \
    --data-path /mnt/datasets/ILSVRC/Data/CLS-LOC/ \
    --log-path /mnt/log/log/DNE \
    --output-basedir /mnt/log/ckpt/DNE --extra-dim 224 --extra-heads 1
