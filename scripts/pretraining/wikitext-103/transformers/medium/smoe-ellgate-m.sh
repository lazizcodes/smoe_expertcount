
args="
--data /cm/shared/stefannvkp/language_modeling/wikitext-103 \
--base_arch transformer \
--architecture sgsgsgsgsEsE \
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
--checkpoint /home/stefannvkp/smoe/checkpoints/pretraining/wikitext103/ellgate-intra-invert-ggggEE-m.pt \
--intra-layer \
--root-invert \
--show-gate-w-stats \
"
echo "Training ..."
CUDA_VISIBLE_DEVICES='0,5,2,3' python -m torch.distributed.launch --master_port 7891 --nproc_per_node=4 --use_env train.py $args


echo "Eval ..."
CUDA_VISIBLE_DEVICES='0,5,2,3' python -m torch.distributed.launch --master_port 7891 --nproc_per_node=4 --use_env train.py $args --resume --full-eval-mode


# for i in 2 3 4 5
#   do
#     echo "Computing Load Balance ..."
#     mkdir -p /home/stefannvkp/Mattention/smoe/checkpoints/
#     args="
#     --data ../wikitext103/lmtool-fwms/data/wikitext-103/ \
#     --base_arch transformer \
#     --architecture sgsgsEsEsEsE \
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
#     --checkpoint /home/stefannvkp/Mattention/smoe/checkpoints/smoe-ellgate-m.pt \
#     --compute_load_balance \
#     --full-eval-mode \
#     --resume
#     "
#     CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 1633 --nproc_per_node=4 --use_env train.py $args
# done