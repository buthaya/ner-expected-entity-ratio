# Run EER on the EE annotated docs, dropping all unannotated docs

RUN_JUPYTER=true
RUN_TENSORBOARD=true
DATASET_LABEL=ee
METHOD_LABEL=eer-short
TRAIN_SUFFIX=_P-1000
BASE_CONFIG=experiments/supervised_tagger_short.jsonnet
ASSUME_COMPLETE="false"

RANDOM_SEED=0
DROPOUT=0.2
LR=2e-5
NUM_EPOCHS=50 # run longer because so few docs
PRIOR_TYPE=null
PRIOR_WEIGHT=10.0


# #  Conll english
# # tacl-eer_eng-c_ee_eer-short-0-30
# bash scripts/run_remote_ml_experiment.sh\
#  PUBLIC_IP=3.91.3.80\
#  PRIVATE_IP=172.31.10.198\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=eng-c\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL-0-30\
#  BASE_CONFIG=$BASE_CONFIG\
#  ASSUME_COMPLETE=$ASSUME_COMPLETE\
#  LANG_DIR=data/conll2003/eng\
#  TRAIN_DATA=entity.train-docs$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=entity.dev-docs.jsonl\
#  TEST_DATA=entity.test-docs.jsonl\
#  VOCAB_PATH=data/conll2003/roberta-entity.vocab\
#  MODEL_NAME=/gpfsdswork/dataset/HuggingFace_Models/roberta-base\
#  PAD_TOKEN="<pad>"\
#  OOV_TOKEN="<unk>"\
#  BATCH_SIZE=15\
#  VALIDATION_BATCH_SIZE=15\
#  RANDOM_SEED=$RANDOM_SEED\
#  DROPOUT=$DROPOUT\
#  LR=$LR\
#  NUM_EPOCHS=$NUM_EPOCHS\
#  PRIOR_WEIGHT=$PRIOR_WEIGHT\
#  ENTITY_RATIO=0.15\
#  ENTITY_RATIO_MARGIN=0.15


# # Conll english
# # tacl-eer_eng-c_ee_eer-short-15-15
# bash scripts/run_remote_ml_experiment.sh\
#  PUBLIC_IP=3.239.39.5\
#  PRIVATE_IP=172.31.8.138\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=eng-c\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL-15-15\
#  BASE_CONFIG=$BASE_CONFIG\
#  ASSUME_COMPLETE=$ASSUME_COMPLETE\
#  LANG_DIR=data/conll2003/eng\
#  TRAIN_DATA=entity.train-docs$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=entity.dev-docs.jsonl\
#  TEST_DATA=entity.test-docs.jsonl\
#  VOCAB_PATH=data/conll2003/roberta-entity.vocab\
#  MODEL_NAME=/gpfsdswork/dataset/HuggingFace_Models/roberta-base\
#  PAD_TOKEN="<pad>"\
#  OOV_TOKEN="<unk>"\
#  BATCH_SIZE=15\
#  VALIDATION_BATCH_SIZE=15\
#  RANDOM_SEED=$RANDOM_SEED\
#  DROPOUT=$DROPOUT\
#  LR=$LR\
#  NUM_EPOCHS=$NUM_EPOCHS\
#  PRIOR_WEIGHT=$PRIOR_WEIGHT\
#  ENTITY_RATIO=0.15\
#  ENTITY_RATIO_MARGIN=0.0




# # Conll english
# # tacl-eer_eng-c_ee_eer-short-23-23
# bash scripts/run_remote_ml_experiment.sh\
#  PUBLIC_IP=3.230.171.236\
#  PRIVATE_IP=172.31.2.121\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=eng-c\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL-23-23\
#  BASE_CONFIG=$BASE_CONFIG\
#  ASSUME_COMPLETE=$ASSUME_COMPLETE\
#  LANG_DIR=data/conll2003/eng\
#  TRAIN_DATA=entity.train-docs$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=entity.dev-docs.jsonl\
#  TEST_DATA=entity.test-docs.jsonl\
#  VOCAB_PATH=data/conll2003/roberta-entity.vocab\
#  MODEL_NAME=/gpfsdswork/dataset/HuggingFace_Models/roberta-base\
#  PAD_TOKEN="<pad>"\
#  OOV_TOKEN="<unk>"\
#  BATCH_SIZE=15\
#  VALIDATION_BATCH_SIZE=15\
#  RANDOM_SEED=$RANDOM_SEED\
#  DROPOUT=$DROPOUT\
#  LR=$LR\
#  NUM_EPOCHS=$NUM_EPOCHS\
#  PRIOR_WEIGHT=$PRIOR_WEIGHT\
#  ENTITY_RATIO=0.2325\
#  ENTITY_RATIO_MARGIN=0.0



# # Conll english
# # tacl-eer_eng-c_ee_eer-short-0-10
# bash scripts/run_remote_ml_experiment.sh\
#  PUBLIC_IP=3.219.240.56\
#  PRIVATE_IP=172.31.7.75\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=eng-c\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL-0-10\
#  BASE_CONFIG=$BASE_CONFIG\
#  ASSUME_COMPLETE=$ASSUME_COMPLETE\
#  LANG_DIR=data/conll2003/eng\
#  TRAIN_DATA=entity.train-docs$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=entity.dev-docs.jsonl\
#  TEST_DATA=entity.test-docs.jsonl\
#  VOCAB_PATH=data/conll2003/roberta-entity.vocab\
#  MODEL_NAME=/gpfsdswork/dataset/HuggingFace_Models/roberta-base\
#  PAD_TOKEN="<pad>"\
#  OOV_TOKEN="<unk>"\
#  BATCH_SIZE=15\
#  VALIDATION_BATCH_SIZE=15\
#  RANDOM_SEED=$RANDOM_SEED\
#  DROPOUT=$DROPOUT\
#  LR=$LR\
#  NUM_EPOCHS=$NUM_EPOCHS\
#  PRIOR_WEIGHT=$PRIOR_WEIGHT\
#  ENTITY_RATIO=0.05\
#  ENTITY_RATIO_MARGIN=0.05



# Conll english
# tacl-eer_eng-c_ee_eer-short-20-30
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=18.215.189.20\
 PRIVATE_IP=172.31.10.198\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=eng-c\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL-20-30\
 BASE_CONFIG=$BASE_CONFIG\
 ASSUME_COMPLETE=$ASSUME_COMPLETE\
 LANG_DIR=data/conll2003/eng\
 TRAIN_DATA=entity.train-docs$TRAIN_SUFFIX.jsonl\
 DEV_DATA=entity.dev-docs.jsonl\
 TEST_DATA=entity.test-docs.jsonl\
 VOCAB_PATH=data/conll2003/roberta-entity.vocab\
 MODEL_NAME=/gpfsdswork/dataset/HuggingFace_Models/roberta-base\
 PAD_TOKEN="<pad>"\
 OOV_TOKEN="<unk>"\
 BATCH_SIZE=15\
 VALIDATION_BATCH_SIZE=15\
 RANDOM_SEED=$RANDOM_SEED\
 DROPOUT=$DROPOUT\
 LR=$LR\
 NUM_EPOCHS=$NUM_EPOCHS\
 PRIOR_WEIGHT=$PRIOR_WEIGHT\
 ENTITY_RATIO=0.25\
 ENTITY_RATIO_MARGIN=0.05




# Conll english
# tacl-eer_eng-c_ee_eer-short-30-30
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=3.228.19.193\
 PRIVATE_IP=172.31.8.138\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=eng-c\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL-30-30\
 BASE_CONFIG=$BASE_CONFIG\
 ASSUME_COMPLETE=$ASSUME_COMPLETE\
 LANG_DIR=data/conll2003/eng\
 TRAIN_DATA=entity.train-docs$TRAIN_SUFFIX.jsonl\
 DEV_DATA=entity.dev-docs.jsonl\
 TEST_DATA=entity.test-docs.jsonl\
 VOCAB_PATH=data/conll2003/roberta-entity.vocab\
 MODEL_NAME=/gpfsdswork/dataset/HuggingFace_Models/roberta-base\
 PAD_TOKEN="<pad>"\
 OOV_TOKEN="<unk>"\
 BATCH_SIZE=15\
 VALIDATION_BATCH_SIZE=15\
 RANDOM_SEED=$RANDOM_SEED\
 DROPOUT=$DROPOUT\
 LR=$LR\
 NUM_EPOCHS=$NUM_EPOCHS\
 PRIOR_WEIGHT=$PRIOR_WEIGHT\
 ENTITY_RATIO=0.30\
 ENTITY_RATIO_MARGIN=0.0
