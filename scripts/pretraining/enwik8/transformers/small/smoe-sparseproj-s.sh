#mkdir -p checkpoints/enwik8/transformers-s/smoe

args="
--data ../lfs/enwik8 \
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
--checkpoint checkpoints/pretraining/enwik8/smoe-sparseproj-noexpertnorm-sqrtw-s.pt \
--show-sparse-w-stats \
--distributed \
"

echo "Training ..."
CUDA_VISIBLE_DEVICES='6,7' python -m torch.distributed.launch --master_port 10233 --nproc_per_node=2 --use_env train.py $args


echo "Eval ..."
CUDA_VISIBLE_DEVICES='6,7' python -m torch.distributed.launch --master_port 10233 --nproc_per_node=2 --use_env train.py $args --resume --full-eval-mode
