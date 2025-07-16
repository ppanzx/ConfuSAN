export CUDA_VISIBLE_DEVICES=1,2

DATASET_NAME='f30k'
DATASET_ROOT="/home/panzx/dataset/CrossModalRetrieval"
DATA_PATH=${DATASET_ROOT}/${DATASET_NAME}
VOCAB_PATH=${DATASET_ROOT}/"vocab"
BUTD_WEIGHT_PATH="/home/panzx/dataset/dependency/ckpt/pretrained"

SAVE_PATH='./scripts/runs/f30k_res152_256x256_bigru_gcn_chan'

echo "Ablation of average pooling with grid featrues of resnet152 save in: "${SAVE_PATH}
python3 train.py \
  --data_path=${DATA_PATH} --data_name=${DATASET_NAME}  --vocab_path=${VOCAB_PATH}\
  --logger_name=${SAVE_PATH}/log --model_name=${SAVE_PATH} \
  --num_epochs=15 --lr_update=10 --learning_rate=5e-4 --workers=8 \
  --text_enc_type=bigru --precomp_enc_type=backbone --backbone_source=imagenet_res152 \
  --vse_mean_warmup_epochs=1 --backbone_warmup_epochs=0 --embedding_warmup_epochs=0 \
  --optim=adam --visual_lr_factor=0.01 --text_lr_factor=0.1 \
  --log_step=200 --batch_size=128 \
  --backbone_path=${BUTD_WEIGHT_PATH}/resnet152-394f9c45.pth \
  --mask --visual_mask_ratio=0.2 --wemb_type=glove \
  --aggr_type=hichan --alpha=0.02 --belta=3 \
  --criterion=MixLoss --temperature=0.01 --margin=0.05 --seed=2024

python3 ./eval.py --dataset=${DATASET_NAME} --model_path=${SAVE_PATH}/model_best.pth --data_path=${DATA_PATH}

exit

SAVE_PATH='./scripts/runs/coco_res152_256x256_bigru_gcn_chan'
DATASET_NAME='coco'
DATA_PATH=${DATASET_ROOT}/${DATASET_NAME}
python3 train.py \
  --data_path=${DATA_PATH} --data_name=${DATASET_NAME}  --vocab_path=${VOCAB_PATH}\
  --logger_name=${SAVE_PATH}/log --model_name=${SAVE_PATH} \
  --num_epochs=15 --lr_update=10 --learning_rate=5e-4 --workers=8 \
  --text_enc_type=bigru --precomp_enc_type=backbone  --backbone_source=imagenet_res152 \
  --vse_mean_warmup_epochs=1 --backbone_warmup_epochs=0 --embedding_warmup_epochs=0 \
  --optim=adam --visual_lr_factor=0.01 --text_lr_factor=0.1 \
  --log_step=200 --batch_size=128 \
  --backbone_path=${BUTD_WEIGHT_PATH}/resnet152-394f9c45.pth \
  --mask --visual_mask_ratio=0.2 --wemb_type=glove \
  --aggr_type=hichan --alpha=0.02 --belta=3 \
  --criterion=MixLoss --temperature=0.01 --margin=0.05 --seed=2024

python3 ./eval.py --dataset=${DATASET_NAME} --model_path=${SAVE_PATH}/model_best.pth --data_path=${DATA_PATH}
