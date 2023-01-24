set -e

GPUS=2,3
NB_COMMA=`echo ${GPUS} | tr -cd , | wc -c`
NB_GPUS=$((${NB_COMMA} + 1))
PORT=$((9000 + RANDOM % 1000))

#shift

echo "Launching exp on $GPUS..."
CUDA_VISIBLE_DEVICES=${GPUS} python -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPUS} main.py \
--options options/data/imagenet100_50-10.yaml options/data/imagenet100_order1.yaml options/model/imagenet_dytox.yaml \
    --name dytox_imgnet100_16heads_P32_b50 \
    --data-path /gpu6_ssd/zhiyuan/datasets/ILSVRC/2012/ \
    --output-basedir /data8/zhiyuan/dytox/ckpt/ --extra-dim 128 --extra-heads 4
    # --resume /data8/zhiyuan/dytox/ckpt/22-08-10_dytox_imgnet100_15heads_b50_1 --start-task 1
