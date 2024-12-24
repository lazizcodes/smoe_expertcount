mkdir -p checkpoints/
args="
--data /cm/shared/stefannvkp/language_modeling/wikitext-103 \
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
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 4000 \
--niter 80 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--mu 0.2 \
--gamma 1.25 \
--checkpoint /home/stefannvkp/Mattention/smoe_expertcount/checkpoints/pretraining/wikitext103/smoe.pt \
"
echo "Training ..."
CUDA_VISIBLE_DEVICES='7' python -m torch.distributed.launch --master_port 10233 --nproc_per_node=1 --use_env train.py $args


# echo "Eval ..."
# CUDA_VISIBLE_DEVICES='2,3,4,5' python -m torch.distributed.launch --master_port 10233 --nproc_per_node=4 --use_env train.py $args --resume --full-eval-mode --wt103_attack


# for i in 0 1 2 3 4 5
#   do
#     echo "Computing Load Balance ..."
#     mkdir -p /home/stefannvkp/Mattention/smoe/checkpoints/
#     args="
#     --data ../wikitext103/lmtool-fwms/data/wikitext-103/ \
#     --base_arch transformer \
#     --architecture sgsgsgsgsgsg \
#     --gate_name smoe \
#     --nlayers 6 \
#     --hid-sz 352 \
#     --inner-hid-sz 352 \
#     --nheads 8 \
#     --block-sz 512 \
#     --attn-span 1024 \
#     --dropout 0.1 \
#     --load_balance 0.01 \
#     --optim adam \
#     --lr 0.0007 \
#     --lr-warmup 4000 \
#     --niter 80 \
#     --batch-sz 48 \
#     --batch-split 2 \
#     --nbatches 1000 \
#     --distributed \
#     --mu 0.2 \
#     --gamma 1.25 \
#     --layer-n $i \
#     --checkpoint /home/stefannvkp/Mattention/smoe/checkpoints/smoe.pt \
#     --compute_load_balance \
#     --resume \
#     --full-eval-mode
#     "
#     CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 1633 --nproc_per_node=4 --use_env train.py $args
# done

#bsize should be 48. Set to 12 for debug