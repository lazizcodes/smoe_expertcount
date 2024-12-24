args="
--data /home/stefannvkp/text_finetune \
--data_name banking77 \
--base_arch transformer \
--architecture sgsgsEsEsEsE \
--gate_name smoe \
--nlayers 6 \
--hid-sz 264 \
--inner-hid-sz 264 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.00 \
--optim adam \
--lr 0.00001 \
--lr-warmup 0 \
--niter 50 \
--batch-sz 16 \
--batch-split 2 \
--nbatches 1000 \
--checkpoint checkpoints/finetuning/banking77/smoe-ellgate-s.pt \
--pretrained_weight checkpoints/pretraining/text8/smoe-ellgate-s.pt \
--distributed \
"

echo "Training ..."
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --master_port 1066 --nproc_per_node=2 --use_env finetune_train.py $args

