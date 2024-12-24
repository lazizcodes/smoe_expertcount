#for i in 1 2 3 4 5
for i in 0
  do
    echo "Eval ..."
    mkdir -p /home/stefannvkp/Mattention/smoe/checkpoints/
    args="
    --data ../wikitext103/lmtool-fwms/data/wikitext-103/ \
    --base_arch transformer \
    --architecture sgegegegegeg \
    --gate_name smoe \
    --nlayers 6 \
    --hid-sz 352 \
    --inner-hid-sz 352 \
    --nheads 8 \
    --block-sz 512 \
    --attn-span 1024 \
    --dropout 0.1 \
    --load_balance 0.01 \
    --optim adam \
    --lr 0.0007 \
    --lr-warmup 4000 \
    --niter 80 \
    --batch-sz 48 \
    --batch-split 2 \
    --nbatches 1000 \
    --distributed \
    --mu 0.2 \
    --gamma 1.25 \
    --layer-n $i \
    --checkpoint /home/stefannvkp/Mattention/smoe/checkpoints/smoe-elliptical.pt \
    --full-eval-mode \
    --resume \
    "
    CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 10233 --nproc_per_node=4 --use_env train.py $args
done