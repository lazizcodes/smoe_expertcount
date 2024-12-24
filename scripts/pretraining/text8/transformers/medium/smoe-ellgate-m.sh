args="
--data ../lfs/text8/text8 \
--base_arch transformer \
--architecture sgsgsEsEsEsEsEsE \
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
--checkpoint checkpoints/pretraining/text8/smoe-ellgate-m.pt \
--distributed \
"

echo "Training ..."
CUDA_VISIBLE_DEVICES='3,4,6,7' python -m torch.distributed.launch --master_port 1223 --nproc_per_node=4 --use_env train.py $args --resume


echo "Eval ..."
CUDA_VISIBLE_DEVICES='3,4,6,7' python -m torch.distributed.launch --master_port 1223 --nproc_per_node=4 --use_env train.py $args --resume --full-eval-mode
