export CUDA_VISIBLE_DEVICES=0,1,2
videos_dir="/home/panzx/dataset/VideoDatasets/Didemo"

torchrun --nproc_per_node=3 train.py \
    --do_train 1 --workers 16 --n_display 50 --epochs 5 \
    --lr 1e-4 --coef_lr 1e-3 --batch_size 42 --batch_size_val 180 \
    --anno_path=${videos_dir}/ \
    --video_path=${videos_dir}/Compressed_Videos \
    --datatype didemo --max_words 64 --max_frames 64 --video_framerate 1 \
    --base_encoder ViT-B/32 --agg_module seqTransf --interaction sparsemax \
    --output_dir ckpts/ckpt_didemo_seqtranf_sparsemax