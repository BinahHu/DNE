set -e

GPUS=0,1
NB_COMMA=`echo ${GPUS} | tr -cd , | wc -c`
NB_GPUS=$((${NB_COMMA} + 1))
PORT=$((9000 + RANDOM % 1000))

#shift

echo "Launching exp on $GPUS..."
CUDA_VISIBLE_DEVICES=${GPUS} python -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPUS} main.py \
--options options/data/cifar100_50-10.yaml options/data/cifar100_order1.yaml options/model/cifar_dytox.yaml \
    --name dytox_test \
    --data-path datasets/CIFAR100/ \
    --output-basedir ckpt/ --extra-dim 32 --extra-heads 1
    #--resume /home/zhiyuan/dytox/ckpt/22-07-13_dytox_split_v2_linear_proj_nofix_attn_b50_1 --start-task 5 --epochs 1
