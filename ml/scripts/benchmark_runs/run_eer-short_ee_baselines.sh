# Run EER on the EE annotated docs, dropping all unannotated docs

RUN_JUPYTER=false
RUN_TENSORBOARD=false
DATASET_LABEL=ee
METHOD_LABEL=eer-short
TRAIN_SUFFIX=_P-1000
BASE_CONFIG=experiments/supervised_tagger_short.jsonnet
ASSUME_COMPLETE="false"

RANDOM_SEED=0
DROPOUT=0.2
LR=2e-5
NUM_EPOCHS=50 # run longer because so few docs
PRIOR_TYPE="eer-exact"
PRIOR_WEIGHT=10.0
ENTITY_RATIO=0.15
ENTITY_RATIO_MARGIN=0.05


# # Conll english
# # tacl-eer_eng-c_ee_eer-short
# bash scripts/run_remote_ml_experiment.sh\
#  PUBLIC_IP=18.204.243.164\
#  PRIVATE_IP=172.31.5.232\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=eng-c\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
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
#  PRIOR_TYPE=$PRIOR_TYPE\
#  PRIOR_WEIGHT=$PRIOR_WEIGHT\
#  ENTITY_RATIO=$ENTITY_RATIO\
#  ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN


# # Conll german
# # tacl-eer_deu_ee_eer-short
# bash scripts/run_remote_ml_experiment.sh\
#  PUBLIC_IP=3.238.56.83\
#  PRIVATE_IP=172.31.4.94\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=deu\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  BASE_CONFIG=$BASE_CONFIG\
#  ASSUME_COMPLETE=$ASSUME_COMPLETE\
#  LANG_DIR=data/conll2003/deu\
#  TRAIN_DATA=entity.train-docs$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=entity.dev-docs.jsonl\
#  TEST_DATA=entity.test-docs.jsonl\
#  VOCAB_PATH=data/conll2003/mbert-entity.vocab\
#  MODEL_NAME=bert-base-multilingual-cased\
#  PAD_TOKEN="[PAD]"\
#  OOV_TOKEN="[UNK]"\
#  BATCH_SIZE=14\
#  VALIDATION_BATCH_SIZE=14\
#  RANDOM_SEED=$RANDOM_SEED\
#  DROPOUT=$DROPOUT\
#  LR=$LR\
#  NUM_EPOCHS=$NUM_EPOCHS\
#  PRIOR_TYPE=$PRIOR_TYPE\
#  PRIOR_WEIGHT=$PRIOR_WEIGHT\
#  ENTITY_RATIO=$ENTITY_RATIO\
#  ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN


# # Conll spanish
# # tacl-eer_esp_ee_eer-short
# bash scripts/run_remote_ml_experiment.sh\
#  PUBLIC_IP=34.200.235.116\
#  PRIVATE_IP=172.31.15.79\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=esp\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  BASE_CONFIG=$BASE_CONFIG\
#  ASSUME_COMPLETE=$ASSUME_COMPLETE\
#  LANG_DIR=data/conll2003/esp\
#  TRAIN_DATA=entity.train-docs$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=entity.dev-docs.jsonl\
#  TEST_DATA=entity.test-docs.jsonl\
#  VOCAB_PATH=data/conll2003/mbert-entity.vocab\
#  MODEL_NAME=bert-base-multilingual-cased\
#  PAD_TOKEN="[PAD]"\
#  OOV_TOKEN="[UNK]"\
#  BATCH_SIZE=14\
#  VALIDATION_BATCH_SIZE=14\
#  RANDOM_SEED=$RANDOM_SEED\
#  DROPOUT=$DROPOUT\
#  LR=$LR\
#  NUM_EPOCHS=$NUM_EPOCHS\
#  PRIOR_TYPE=$PRIOR_TYPE\
#  PRIOR_WEIGHT=$PRIOR_WEIGHT\
#  ENTITY_RATIO=$ENTITY_RATIO\
#  ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN


# # Conll dutch
# # tacl-eer_ned_ee_eer-short
# bash scripts/run_remote_ml_experiment.sh\
#  PUBLIC_IP=3.91.34.64\
#  PRIVATE_IP=172.31.5.25\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=ned\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  BASE_CONFIG=$BASE_CONFIG\
#  ASSUME_COMPLETE=$ASSUME_COMPLETE\
#  LANG_DIR=data/conll2003/ned\
#  TRAIN_DATA=entity.train-docs$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=entity.dev-docs.jsonl\
#  TEST_DATA=entity.test-docs.jsonl\
#  VOCAB_PATH=data/conll2003/mbert-entity.vocab\
#  MODEL_NAME=bert-base-multilingual-cased\
#  PAD_TOKEN="[PAD]"\
#  OOV_TOKEN="[UNK]"\
#  BATCH_SIZE=14\
#  VALIDATION_BATCH_SIZE=14\
#  RANDOM_SEED=$RANDOM_SEED\
#  DROPOUT=$DROPOUT\
#  LR=$LR\
#  NUM_EPOCHS=$NUM_EPOCHS\
#  PRIOR_TYPE=$PRIOR_TYPE\
#  PRIOR_WEIGHT=$PRIOR_WEIGHT\
#  ENTITY_RATIO=$ENTITY_RATIO\
#  ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN


# Onto english
# tacl-eer_eng-o_ee_eer-short
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=3.208.94.234\
 PRIVATE_IP=172.31.10.164\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=eng-o\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 BASE_CONFIG=$BASE_CONFIG\
 ASSUME_COMPLETE=$ASSUME_COMPLETE\
 LANG_DIR=data/ontonotes5/processed_docs/english\
 TRAIN_DATA=train$TRAIN_SUFFIX.jsonl\
 DEV_DATA=dev.jsonl\
 TEST_DATA=test.jsonl\
 VOCAB_PATH=data/ontonotes5/processed_docs/roberta-entity.vocab\
 MODEL_NAME=/gpfsdswork/dataset/HuggingFace_Models/roberta-base\
 PAD_TOKEN="<pad>"\
 OOV_TOKEN="<unk>"\
 BATCH_SIZE=2\
 VALIDATION_BATCH_SIZE=1\
 RANDOM_SEED=$RANDOM_SEED\
 DROPOUT=$DROPOUT\
 LR=$LR\
 NUM_EPOCHS=$NUM_EPOCHS\
 PRIOR_TYPE=$PRIOR_TYPE\
 PRIOR_WEIGHT=$PRIOR_WEIGHT\
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN


# Onto chinese
# tacl-eer_chi_ee_eer-short
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=3.235.154.36\
 PRIVATE_IP=172.31.2.41\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=chi\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 BASE_CONFIG=$BASE_CONFIG\
 ASSUME_COMPLETE=$ASSUME_COMPLETE\
 LANG_DIR=data/ontonotes5/processed_docs/chinese\
 TRAIN_DATA=train$TRAIN_SUFFIX.jsonl\
 DEV_DATA=dev.jsonl\
 TEST_DATA=test.jsonl\
 VOCAB_PATH=data/ontonotes5/processed_docs/mbert-entity.vocab\
 MODEL_NAME=bert-base-multilingual-cased\
 PAD_TOKEN="[PAD]"\
 OOV_TOKEN="[UNK]"\
 BATCH_SIZE=2\
 VALIDATION_BATCH_SIZE=1\
 RANDOM_SEED=$RANDOM_SEED\
 DROPOUT=$DROPOUT\
 LR=$LR\
 NUM_EPOCHS=$NUM_EPOCHS\
 PRIOR_TYPE=$PRIOR_TYPE\
 PRIOR_WEIGHT=$PRIOR_WEIGHT\
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN



# # Onto arabic
# # tacl-eer_ara_ee_eer-short
# bash scripts/run_remote_ml_experiment.sh\
#  PUBLIC_IP=54.157.107.213\
#  PRIVATE_IP=172.31.8.225\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=ara\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  BASE_CONFIG=$BASE_CONFIG\
#  ASSUME_COMPLETE=$ASSUME_COMPLETE\
#  LANG_DIR=data/ontonotes5/processed_docs/arabic\
#  TRAIN_DATA=train$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=dev.jsonl\
#  TEST_DATA=test.jsonl\
#  VOCAB_PATH=data/ontonotes5/processed_docs/mbert-entity.vocab\
#  MODEL_NAME=bert-base-multilingual-cased\
#  PAD_TOKEN="[PAD]"\
#  OOV_TOKEN="[UNK]"\
#  BATCH_SIZE=2\
#  VALIDATION_BATCH_SIZE=1\
#  RANDOM_SEED=$RANDOM_SEED\
#  DROPOUT=$DROPOUT\
#  LR=$LR\
#  NUM_EPOCHS=$NUM_EPOCHS\
#  PRIOR_TYPE=$PRIOR_TYPE\
#  PRIOR_WEIGHT=$PRIOR_WEIGHT\
#  ENTITY_RATIO=$ENTITY_RATIO\
#  ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN
