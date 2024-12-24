args="
--data /cm/shared/stefannvkp/language_modeling/text_finetune/imdb/ \
--data_name imdb \
--base_arch transformer \
--architecture rgegegegegegegeg \
--gate_name smoe \
--nlayers 8 \
--hid-sz 352 \
--inner-hid-sz 352 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.00 \
--optim adam \
--lr 0.0001 \
--lr-warmup 0 \
--niter 5 \
--batch-sz 4 \
--batch-split 2 \
--nbatches 1000 \
--checkpoint checkpoints/finetuning/imdb/medium/ellattention-smoe-m.pt
--pretrained_weight checkpoints/pretraining/enwik8/ellattention-smoe-m.pt \
--distributed \
"

echo "Training ..."
CUDA_VISIBLE_DEVICES='7' python -m torch.distributed.launch --master_port 9995 --nproc_per_node=1 --use_env finetune_train.py $args


# echo "Eval ..."
# CUDA_VISIBLE_DEVICES='0,1,2,7' python -m torch.distributed.launch --master_port 1903 --nproc_per_node=4 --use_env train.py $args --resume --full-eval-mode
