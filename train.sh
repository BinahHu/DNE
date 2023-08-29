set -e

GPUS=0,1,2,3
NB_COMMA=`echo ${GPUS} | tr -cd , | wc -c`
NB_GPUS=$((${NB_COMMA} + 1))
PORT=$((9000 + RANDOM % 1000))

#shift

echo "Launching exp on $GPUS..."
CUDA_VISIBLE_DEVICES=${GPUS} python -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPUS} main.py \
--options options/data/cifar100_50-10.yaml options/data/cifar100_order1.yaml options/model/cifar_dne.yaml \
    --name dne_mlp_dense_only_cifar100_b50_10 \
    --data-path /mnt/datasets/CIFAR100/ \
    --output-basedir /mnt/log/ckpt/DNE \
    --output-basedir ckpt/ --extra-dim 32 --extra-heads 1
