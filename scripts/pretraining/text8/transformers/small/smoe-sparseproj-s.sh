args="
--data ../lfs/text8/text8 \
--base_arch transformer \
--architecture sgspspspspsp \
--gate_name smoe \
--nlayers 6 \
--hid-sz 264 \
--inner-hid-sz 264 \
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
--show-sparse-w-stats \
--checkpoint checkpoints/pretraining/text8/smoe-sparseproj-s.pt \
--distributed \
"

echo "Training ..."
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 1523 --nproc_per_node=4 --use_env train.py $args


echo "Eval ..."
CUDA_VISIBLE_DEVICES='0,6,7' python -m torch.distributed.launch --master_port 1523 --nproc_per_node=3 --use_env train.py $args --resume --full-eval-mode
