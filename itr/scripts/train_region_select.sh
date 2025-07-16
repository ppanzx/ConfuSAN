export CUDA_VISIBLE_DEVICES=0,2

DATASET_NAME='f30k'
DATASET_ROOT="/home/panzx/dataset/CrossModalRetrieval"
DATA_PATH=${DATASET_ROOT}/${DATASET_NAME}
VOCAB_PATH=${DATASET_ROOT}/"vocab"
BUTD_WEIGHT_PATH=${DATASET_ROOT}/"weights"

## wasserstain with region features
SAVE_PATH='./scripts/runs/debug'
python3 ./train.py \
  --data_path=${DATA_PATH} --data_name=${DATASET_NAME} --text_enc_type=bigru \
  --vocab_path=${VOCAB_PATH} --logger_name=${SAVE_PATH}/log --model_name=${SAVE_PATH} \
  --num_epochs=15 --lr_update=10 --learning_rate=5e-4 --precomp_enc_type=gcn --workers=5 \
  --log_step=200 --embed_size=1024 --vse_mean_warmup_epochs=20 --batch_size=384 \
  --aggr_type=select --KNN=1 --alpha=4 \
  --mask --visual_mask_ratio=0.2 --wemb_type=glove \
  --criterion=MixLoss --temperature=0.01 --margin=0.05 --seed=2023

python3 ./eval.py --dataset=${DATASET_NAME} --model_path=${SAVE_PATH}/model_best.pth --data_path=${DATA_PATH}