mkdir -p checkpoints/finetune
args="
--data /home/stefannvkp/text_finetune \
--data_name imdb \
--base_arch transformer \
--architecture sgsgsgsgsgsg \
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
--lr 0.0001 \
--lr-warmup 0 \
--niter 5 \
--batch-sz 4 \
--batch-split 2 \
--nbatches 1000 \
--checkpoint checkpoints/finetuning/imdb/smoe-s.pt \
--pretrained_weight checkpoints/pretraining/enwik8/smoe-s.pt \
--distributed \
"

echo "Training ..."
CUDA_VISIBLE_DEVICES='7' python -m torch.distributed.launch --master_port 10233 --nproc_per_node=1 --use_env finetune_train.py $args

# echo 'Evaluation ...'
# CUDA_VISIBLE_DEVICES='6,7' python -m torch.distributed.launch --master_port 10233 --nproc_per_node=2 --use_env finetune_train.py $args --full-eval-mode
