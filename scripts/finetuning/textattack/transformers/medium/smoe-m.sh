args="
--data /home/stefannvkp/Mattention/wikitext103/lmtool-fwms/data/wikitext-103 \
--data_name wt103 \
--wt103_attack
--base_arch transformer \
--architecture sgsgsgsgsgsg \
--gate_name smoe \
--nlayers 6 \
--hid-sz 352 \
--inner-hid-sz 352 \
--nheads 8 \
--block-sz 512 \
--attn-span 1024 \
--dropout 0.1 \
--load_balance 0.00 \
--optim adam \
--lr 0.0001 \
--lr-warmup 0 \
--niter 5 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 1000 \
--checkpoint checkpoints/finetuning/textattack/smoe-m.pt \
--pretrained_weight checkpoints/pretraining/wikitext103/smoe.pt \
--distributed \
--full-eval-mode \
--resume \
"
# previous batch size 48

echo "Evaluating ..."
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --master_port 1066 --nproc_per_node=4 --use_env finetune_train.py $args

