args="
--data /cm/shared/stefannvkp/language_modeling/text8 \
--base_arch transformer \
--architecture sgsgsgsgsgsgsgsg \
--gate_name smoe \
--nlayers 8 \
--hid-sz 352 \
--inner-hid-sz 352 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 4000 \
--niter 80 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 1000 \
--checkpoint checkpoints/pretraining/text8/smoe-m.pt \
--distributed \
"

echo "Training ..."
CUDA_VISIBLE_DEVICES='5,6,7' python -m torch.distributed.launch --master_port 1903 --nproc_per_node=3 --use_env train.py $args


echo "Eval ..."
CUDA_VISIBLE_DEVICES='5,6,7' python -m torch.distributed.launch --master_port 1903 --nproc_per_node=3 --use_env train.py $args --resume --full-eval-mode
