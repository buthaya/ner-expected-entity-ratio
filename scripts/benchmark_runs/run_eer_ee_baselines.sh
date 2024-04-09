#!/bin/bash
#SBATCH --job-name=eer_ee_baselines
#SBATCH --output=out_eer_ee_baselines.txt
#SBATCH --error=err_eer_ee_baselines.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10 
#SBATCH --time=10:00:00
#SBATCH -C v100-32g
#SBATCH -A zke@v100

# Navigate to dir
cd $SCRATCH/ner-expected-entity-ratio/
module load python
conda activate ner-eer/
which python 
which pip
pwd

# Remote settings variables
IS_REMOTE=false
RUN_JUPYTER=false
RUN_TENSORBOARD=false

# Experiment Dataset settings
DATASET_LABEL=ee
METHOD_LABEL=eer

# Hyperparameters common for all experiments 
ASSUME_COMPLETE=false
TRAIN_SUFFIX=_P-1000
RANDOM_SEED=0
DROPOUT=0.2
LR=2e-5
NUM_EPOCHS=20
PRIOR_WEIGHT=10.0
ENTITY_RATIO=0.15
ENTITY_RATIO_MARGIN=0.05


# # Conll english
# tacl-eer_eng-c_linear_raw-short
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
#  ENTITY_RATIO=$ENTITY_RATIO\
#  ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN
#  PRIOR_TYPE=null
 

# Conll german
# tacl-eer_deu_ee_eer
bash scripts/run_ml_experiment.sh\
 IS_REMOTE=$IS_REMOTE\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=deu\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 BASE_CONFIG=experiments/supervised_tagger.jsonnet\
 ASSUME_COMPLETE=$ASSUME_COMPLETE\
 LANG_DIR=data/conll2003/deu\
 TRAIN_DATA=entity.train-docs$TRAIN_SUFFIX.jsonl\
 DEV_DATA=entity.dev-docs.jsonl\
 TEST_DATA=entity.test-docs.jsonl\
 VOCAB_PATH=data/conll2003/mbert-entity.vocab\
 MODEL_NAME=/gpfsdswork/dataset/HuggingFace_Models/bert-base-multilingual-cased\
 PAD_TOKEN="[PAD]"\
 OOV_TOKEN="[UNK]"\
 BATCH_SIZE=14\
 VALIDATION_BATCH_SIZE=14\
 RANDOM_SEED=$RANDOM_SEED\
 DROPOUT=$DROPOUT\
 LR=$LR\
 NUM_EPOCHS=$NUM_EPOCHS\
 PRIOR_WEIGHT=$PRIOR_WEIGHT\
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN
#  PRIOR_TYPE=$PRIOR_TYPE



# Conll spanish
# tacl-eer_esp_ee_eer
bash scripts/run_ml_experiment.sh\
 IS_REMOTE=$IS_REMOTE\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=esp\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 BASE_CONFIG=experiments/supervised_tagger.jsonnet\
 ASSUME_COMPLETE=$ASSUME_COMPLETE\
 LANG_DIR=data/conll2003/esp\
 TRAIN_DATA=entity.train-docs$TRAIN_SUFFIX.jsonl\
 DEV_DATA=entity.dev-docs.jsonl\
 TEST_DATA=entity.test-docs.jsonl\
 VOCAB_PATH=data/conll2003/mbert-entity.vocab\
 MODEL_NAME=/gpfsdswork/dataset/HuggingFace_Models/bert-base-multilingual-cased\
 PAD_TOKEN="[PAD]"\
 OOV_TOKEN="[UNK]"\
 BATCH_SIZE=14\
 VALIDATION_BATCH_SIZE=14\
 RANDOM_SEED=$RANDOM_SEED\
 DROPOUT=$DROPOUT\
 LR=$LR\
 NUM_EPOCHS=$NUM_EPOCHS\
 PRIOR_WEIGHT=$PRIOR_WEIGHT\
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN
#  PRIOR_TYPE=$PRIOR_TYPE


# Conll dutch
# tacl-eer_ned_ee_eer
bash scripts/run_ml_experiment.sh\
 IS_REMOTE=$IS_REMOTE\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=ned\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 BASE_CONFIG=experiments/supervised_tagger.jsonnet\
 ASSUME_COMPLETE=$ASSUME_COMPLETE\
 LANG_DIR=data/conll2003/ned\
 TRAIN_DATA=entity.train-docs$TRAIN_SUFFIX.jsonl\
 DEV_DATA=entity.dev-docs.jsonl\
 TEST_DATA=entity.test-docs.jsonl\
 VOCAB_PATH=data/conll2003/mbert-entity.vocab\
 MODEL_NAME=/gpfsdswork/dataset/HuggingFace_Models/bert-base-multilingual-cased\
 PAD_TOKEN="[PAD]"\
 OOV_TOKEN="[UNK]"\
 BATCH_SIZE=14\
 VALIDATION_BATCH_SIZE=14\
 RANDOM_SEED=$RANDOM_SEED\
 DROPOUT=$DROPOUT\
 LR=$LR\
 NUM_EPOCHS=$NUM_EPOCHS\
 PRIOR_WEIGHT=$PRIOR_WEIGHT\
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN
#  PRIOR_TYPE=$PRIOR_TYPE


# Ontonotes english
# tacl-eer_eng-o_ee_eer
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
 PRIOR_WEIGHT=$PRIOR_WEIGHT\
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN
#  PRIOR_TYPE=$PRIOR_TYPE


# Ontonotes chinese
# tacl-eer_chi_ee_eer
bash scripts/run_ml_experiment.sh\
 IS_REMOTE=$IS_REMOTE\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=chi\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 BASE_CONFIG=experiments/supervised_tagger.jsonnet\
 ASSUME_COMPLETE=$ASSUME_COMPLETE\
 LANG_DIR=data/ontonotes5/processed_docs/chinese\
 TRAIN_DATA=train$TRAIN_SUFFIX.jsonl\
 DEV_DATA=dev.jsonl\
 TEST_DATA=test.jsonl\
 VOCAB_PATH=data/ontonotes5/processed_docs/mbert-entity.vocab\
 MODEL_NAME=/gpfsdswork/dataset/HuggingFace_Models/bert-base-multilingual-cased\
 PAD_TOKEN="[PAD]"\
 OOV_TOKEN="[UNK]"\
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


# Ontonotes arabic
# tacl-eer_ara_ee_eer
bash scripts/run_ml_experiment.sh\
 IS_REMOTE=$IS_REMOTE\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=ara\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 BASE_CONFIG=experiments/supervised_tagger.jsonnet\
 ASSUME_COMPLETE=$ASSUME_COMPLETE\
 LANG_DIR=data/ontonotes5/processed_docs/arabic\
 TRAIN_DATA=train$TRAIN_SUFFIX.jsonl\
 DEV_DATA=dev.jsonl\
 TEST_DATA=test.jsonl\
 VOCAB_PATH=data/ontonotes5/processed_docs/mbert-entity.vocab\
 MODEL_NAME=/gpfsdswork/dataset/HuggingFace_Models/bert-base-multilingual-cased\
 PAD_TOKEN="[PAD]"\
 OOV_TOKEN="[UNK]"\
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

