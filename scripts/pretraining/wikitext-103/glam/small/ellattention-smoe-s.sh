
args="
--data /cm/shared/stefannvkp/language_modeling/wikitext-103 \
--base_arch glam \
--architecture rgefegefegef \
--gate_name smoe \
--nlayers 3 \
--hid-sz 144 \
--inner-hid-sz 144 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 3000 \
--niter 60 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 1000 \
--checkpoint checkpoints/pretraining/wikitext-103/glam-s/smoe/ellattention-smoe.pt \
"

echo "Training ..."
CUDA_VISIBLE_DEVICES='5' python -m torch.distributed.launch --master_port 1294 --nproc_per_node=1 --use_env train.py $args


echo "Eval ..."
CUDA_VISIBLE_DEVICES='5' python -m torch.distributed.launch --master_port 1294 --nproc_per_node=1 --use_env train.py $args --resume --full-eval-mode

# orig batch size 48