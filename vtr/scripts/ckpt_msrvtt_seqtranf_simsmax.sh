export CUDA_VISIBLE_DEVICES=1,2
videos_dir="/home/panzx/dataset/VideoDatasets/MSR_VTT"

# Token-wise Interaction
torchrun --nproc_per_node=2 train.py \
    --do_train 1 --workers 8 --n_display 50 --epochs 5 \
    --lr 1e-4 --coef_lr 1e-3 --batch_size 140 --batch_size_val 216 \
    --anno_path=${videos_dir}/msrvtt_data \
    --video_path=${videos_dir}/Compressed_Videos \
    --datatype msrvtt --max_words 24 --max_frames 12 --video_framerate 1 \
    --base_encoder ViT-B/32 --agg_module seqTransf --interaction simsmax \
    --cdcr_alpha1=0.0 --output_dir=ckpts/ckpt_msrvtt_seqtranf_simsmax