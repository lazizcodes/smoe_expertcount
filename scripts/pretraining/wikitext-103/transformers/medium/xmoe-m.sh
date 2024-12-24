mkdir -p /home/stefannvkp/Mattention/smoe/checkpoints/

args="
--data ../wikitext103/lmtool-fwms/data/wikitext-103/ \
--base_arch transformer \
--architecture sgsgsgsgsgsg \
--gate_name xmoe \
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
--checkpoint /home/stefannvkp/Mattention/smoe/checkpoints/trans-xmoe-m.pt \
--distributed \
"

# echo "Training ..."
# CUDA_VISIBLE_DEVICES='1,2,3' python -m torch.distributed.launch --master_port 1113 --nproc_per_node=3 --use_env train.py $args

echo "Evaluation ..."
CUDA_VISIBLE_DEVICES='1,2,3' python -m torch.distributed.launch --master_port 1113 --nproc_per_node=3 --use_env train.py $args --resume --full-eval-mode --compute-rep-collapse