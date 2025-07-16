export CUDA_VISIBLE_DEVICES=2

DATASET_NAME='f30k'
DATASET_ROOT="/home/panzx/dataset/CrossModalRetrieval"
DATA_PATH=${DATASET_ROOT}/${DATASET_NAME}
VOCAB_PATH=${DATASET_ROOT}/"vocab"
BUTD_WEIGHT_PATH=${DATASET_ROOT}/"weights"

SAVE_PATH='./scripts/runs/f30k_clip_select'
python3 train.py \
  --data_path=${DATA_PATH} --data_name=${DATASET_NAME}  --vocab_path=${VOCAB_PATH} \
  --logger_name=${SAVE_PATH}/log --model_name=${SAVE_PATH} --precomp_enc_type=clip --text_enc_type=clip \
  --num_epochs=15 --lr_update=10 --learning_rate=5e-4 --workers=8 --embed_size=512 \
  --vse_mean_warmup_epochs=0 --backbone_warmup_epochs=0 --embedding_warmup_epochs=0 --optim=adam \
  --visual_lr_factor=1e-3 --text_lr_factor=1e-3 --log_step=200 --batch_size=128 \
  --mask --visual_mask_ratio=0 \
  --aggr_type=hichan --coding_type=hichanv2 \
  --criterion=InfoNCELoss --alpha=0
  # --criterion=ContrastiveLoss --margin=0.1
# 

python3 ./eval.py --dataset=${DATASET_NAME} --model_path=${SAVE_PATH}/model_best.pth --data_path=${DATA_PATH}