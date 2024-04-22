#!/bin/bash
#SBATCH --job-name=gold_baselines
#SBATCH --output=out_gold_baselines.txt
#SBATCH --error=err_gold_baselines.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10 
#SBATCH --time=10:00:00
#SBATCH -C v100-32g
#SBATCH -A zke@v100

# Benchmark Experiment: Gold mle baseline on all datasets
RUN_JUPYTER=false
RUN_TENSORBOARD=false
DATASET_LABEL=gold
METHOD_LABEL=mle
ASSUME_COMPLETE="true"
RANDOM_SEED=0
DROPOUT=0.2
LR=2e-5
NUM_EPOCHS=20
PRIOR_TYPE=null
PRIOR_WEIGHT=0.0
ENTITY_RATIO=0.15
ENTITY_RATIO_MARGIN=0.05
IS_REMOTE=false

# # Conll english
# # tacl-eer_eng-c_gold_mle
# bash scripts/run_ml_experiment.sh\
#  IS_REMOTE=$IS_REMOTE\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=eng-c\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  BASE_CONFIG=experiments/supervised_tagger.jsonnet\
#  ASSUME_COMPLETE=$ASSUME_COMPLETE\
#  LANG_DIR=data/conll2003/eng\
#  TRAIN_DATA=entity.train-docs.jsonl\
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
#  ENTITY_RATIO=$ENTITY_RATIO\
#  ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN
# #  PRIOR_TYPE=$PRIOR_TYPE
 


# # Conll german
# # tacl-eer_deu_gold_mle
# bash scripts/run_remote_ml_experiment.sh\
#  IS_REMOTE=$IS_REMOTE\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=deu\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  BASE_CONFIG=experiments/supervised_tagger.jsonnet\
#  ASSUME_COMPLETE=$ASSUME_COMPLETE\
#  LANG_DIR=data/conll2003/deu\
#  TRAIN_DATA=entity.train-docs.jsonl\
#  DEV_DATA=entity.dev-docs.jsonl\
#  TEST_DATA=entity.test-docs.jsonl\
#  VOCAB_PATH=data/conll2003/mbert-entity.vocab\
#  MODEL_NAME=/gpfsdswork/dataset/HuggingFace_Models/bert-base-multilingual-cased\
#  PAD_TOKEN="[PAD]"\
#  OOV_TOKEN="[UNK]"\
#  BATCH_SIZE=14\
#  VALIDATION_BATCH_SIZE=14\
#  RANDOM_SEED=$RANDOM_SEED\
#  DROPOUT=$DROPOUT\
#  LR=$LR\
#  NUM_EPOCHS=$NUM_EPOCHS\
#  PRIOR_WEIGHT=$PRIOR_WEIGHT\
#  ENTITY_RATIO=$ENTITY_RATIO\
#  ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN\
##  PRIOR_TYPE=$PRIOR_TYPE



# # Conll spanish
# # tacl-eer_esp_gold_mle
# bash scripts/run_remote_ml_experiment.sh\
#  IS_REMOTE=$IS_REMOTE\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=esp\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  BASE_CONFIG=experiments/supervised_tagger.jsonnet\
#  ASSUME_COMPLETE=$ASSUME_COMPLETE\
#  LANG_DIR=data/conll2003/esp\
#  TRAIN_DATA=entity.train-docs.jsonl\
#  DEV_DATA=entity.dev-docs.jsonl\
#  TEST_DATA=entity.test-docs.jsonl\
#  VOCAB_PATH=data/conll2003/mbert-entity.vocab\
#  MODEL_NAME=/gpfsdswork/dataset/HuggingFace_Models/bert-base-multilingual-cased\
#  PAD_TOKEN="[PAD]"\
#  OOV_TOKEN="[UNK]"\
#  BATCH_SIZE=14\
#  VALIDATION_BATCH_SIZE=14\
#  RANDOM_SEED=$RANDOM_SEED\
#  DROPOUT=$DROPOUT\
#  LR=$LR\
#  NUM_EPOCHS=$NUM_EPOCHS\
#  PRIOR_WEIGHT=$PRIOR_WEIGHT\
#  ENTITY_RATIO=$ENTITY_RATIO\
#  ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN\
##  PRIOR_TYPE=$PRIOR_TYPE


# # Conll dutch
# # tacl-eer_ned_gold_mle
# bash scripts/run_remote_ml_experiment.sh\
#  IS_REMOTE=$IS_REMOTE\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=ned\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  BASE_CONFIG=experiments/supervised_tagger.jsonnet\
#  ASSUME_COMPLETE=$ASSUME_COMPLETE\
#  LANG_DIR=data/conll2003/ned\
#  TRAIN_DATA=entity.train-docs.jsonl\
#  DEV_DATA=entity.dev-docs.jsonl\
#  TEST_DATA=entity.test-docs.jsonl\
#  VOCAB_PATH=data/conll2003/mbert-entity.vocab\
#  MODEL_NAME=/gpfsdswork/dataset/HuggingFace_Models/bert-base-multilingual-cased\
#  PAD_TOKEN="[PAD]"\
#  OOV_TOKEN="[UNK]"\
#  BATCH_SIZE=14\
#  VALIDATION_BATCH_SIZE=14\
#  RANDOM_SEED=$RANDOM_SEED\
#  DROPOUT=$DROPOUT\
#  LR=$LR\
#  NUM_EPOCHS=$NUM_EPOCHS\
#  PRIOR_WEIGHT=$PRIOR_WEIGHT\
#  ENTITY_RATIO=$ENTITY_RATIO\
#  ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN\
##  PRIOR_TYPE=$PRIOR_TYPE


# Ontonotes english
# tacl-eer_eng-o_gold_mle
bash scripts/run_ml_experiment.sh\
 IS_REMOTE=$IS_REMOTE\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=eng-o\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 BASE_CONFIG=experiments/supervised_tagger.jsonnet\
 ASSUME_COMPLETE=$ASSUME_COMPLETE\
 LANG_DIR=data/ontonotes5/processed_docs/english\
 TRAIN_DATA=train.jsonl\
 DEV_DATA=dev.jsonl\
 TEST_DATA=test.jsonl\
 VOCAB_PATH=data/ontonotes5/processed/roberta-entity.vocab\
 MODEL_NAME=/gpfsdswork/dataset/HuggingFace_Models/roberta-base\
 PAD_TOKEN="<pad>"\
 OOV_TOKEN="<unk>"\
 BATCH_SIZE=2\
 VALIDATION_BATCH_SIZE=1\
 RANDOM_SEED=$RANDOM_SEED\
 DROPOUT=$DROPOUT\
 LR=$LR\
 NUM_EPOCHS=$NUM_EPOCHS\
 PRIOR_WEIGHT=$PRIOR_WEIGHT\
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN
#  PRIOR_TYPE=$PRIOR_TYPE


# # Ontonotes chinese
# # tacl-eer_chi_gold_mle
# bash scripts/run_remote_ml_experiment.sh\
#  IS_REMOTE=$IS_REMOTE\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=chi\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  BASE_CONFIG=experiments/supervised_tagger.jsonnet\
#  ASSUME_COMPLETE=$ASSUME_COMPLETE\
#  LANG_DIR=data/ontonotes5/processed_docs/chinese\
#  TRAIN_DATA=train.jsonl\
#  DEV_DATA=dev.jsonl\
#  TEST_DATA=test.jsonl\
#  VOCAB_PATH=data/ontonotes5/mbert-entity.vocab\
#  MODEL_NAME=/gpfsdswork/dataset/HuggingFace_Models/bert-base-multilingual-cased\
#  PAD_TOKEN="[PAD]"\
#  OOV_TOKEN="[UNK]"\
#  BATCH_SIZE=2\
#  VALIDATION_BATCH_SIZE=1\
#  RANDOM_SEED=$RANDOM_SEED\
#  DROPOUT=$DROPOUT\
#  LR=$LR\
#  NUM_EPOCHS=$NUM_EPOCHS\
#  PRIOR_WEIGHT=$PRIOR_WEIGHT\
#  ENTITY_RATIO=$ENTITY_RATIO\
#  ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN\
##  PRIOR_TYPE=$PRIOR_TYPE


# # Ontonotes arabic
# # tacl-eer_ara_gold_mle
# bash scripts/run_remote_ml_experiment.sh\
#  IS_REMOTE=$IS_REMOTE\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=ara\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  BASE_CONFIG=experiments/supervised_tagger.jsonnet\
#  ASSUME_COMPLETE=$ASSUME_COMPLETE\
#  LANG_DIR=data/ontonotes5/processed_docs/arabic\
#  TRAIN_DATA=train.jsonl\
#  DEV_DATA=dev.jsonl\
#  TEST_DATA=test.jsonl\
#  VOCAB_PATH=data/ontonotes5/mbert-entity.vocab\
#  MODEL_NAME=/gpfsdswork/dataset/HuggingFace_Models/bert-base-multilingual-cased\
#  PAD_TOKEN="[PAD]"\
#  OOV_TOKEN="[UNK]"\
#  BATCH_SIZE=2\
#  VALIDATION_BATCH_SIZE=1\
#  RANDOM_SEED=$RANDOM_SEED\
#  DROPOUT=$DROPOUT\
#  LR=$LR\
#  NUM_EPOCHS=$NUM_EPOCHS\
#  PRIOR_WEIGHT=$PRIOR_WEIGHT\
#  ENTITY_RATIO=$ENTITY_RATIO\
#  ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN\
##  PRIOR_TYPE=$PRIOR_TYPE
