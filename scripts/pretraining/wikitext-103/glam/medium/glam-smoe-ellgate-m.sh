mkdir -p /home/stefannvkp/Mattention/smoe_expertcount/checkpoints/

args="
--data ../wikitext103/lmtool-fwms/data/wikitext-103/ \
--base_arch glam \
--architecture sgsfsgsfsEsfsEsf \
--gate_name smoe \
--nlayers 4 \
--hid-sz 352 \
--inner-hid-sz 352 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.00007 \
--lr-warmup 4000 \
--niter 80 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 1000 \
--checkpoint /home/stefannvkp/Mattention/smoe_expertcount/checkpoints/glam-ellgate-m.pt \
--distributed \
"

echo "Training ..."
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=4 --use_env train.py $args

echo "Evaluation ..."
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=4 --use_env train.py $args --resume --full-eval-mode

# try checking eval mode with batchsize 8 as per Rachel's expertcount branch code