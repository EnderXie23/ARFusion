export CUDA_VISIBLE_DEVICES=1

python train.py \
	--dataset cvprw2020-ade20K-defg \
	--checkpoint_dir checkpoints_bisenet \
	--vis_dir val_out_demo \
	--in_size 384 \
	--max_num_epochs 100 \
	--lr 1e-4 \
	--batch_size 8 \
	--net_G bisenetv2