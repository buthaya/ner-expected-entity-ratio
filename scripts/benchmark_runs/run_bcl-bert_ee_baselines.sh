#!/bin/bash
#SBATCH --job-name=run_bcl-bert_ee_baselines.sh
#SBATCH --output=out_bcl-bert_ee_baselines.txt
#SBATCH --error=err_bcl-bert_ee_baselines%j.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10 
#SBATCH --time=10:00:00
#SBATCH -C v100-32g
#SBATCH -A zke@v100

# Benchmark Experiment: BCL-BERT on EE doc data (the adapted Mayhew 19 code)
RUN_JUPYTER=false
RUN_TENSORBOARD=false
DATASET_LABEL=ee
METHOD_LABEL=bcl-bert
TRAIN_SUFFIX=_P-1000
MODEL_NAME='bert-bcl-ee'
PAD_TOKEN="<pad>"
OOV_TOKEN="<unk>"
BATCH_SIZE=15
RANDOM_SEED=0
DROPOUT=0.2
LR=2e-5
ENTITY_RATIO=0.15
ENTITY_RATIO_MARGIN=0.05
NUM_EPOCHS=50

# # Conll english
# tacl-eer_eng-c_ee_bcl-bert
bash scripts/run_ml_experiment.sh\
 IS_REMOTE=false\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=eng-c\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 LANG_DIR=data/conll2003/eng\
 TRAIN_DATA=entity.train-docs$TRAIN_SUFFIX.jsonl\
 DEV_DATA=entity.dev-docs.jsonl\
 TEST_DATA=entity.test-docs.jsonl\
 VOCAB_PATH=data/conll2003/mayhew-entity.vocab\
 BINARY_VOCAB_PATH=data/conll2003/mayhew-binary-entity.vocab\
 VECTORS_PATH=data/vectors/glove.6B.50d.txt\
 BASE_CONFIG=experiments/supervised_tagger.jsonnet\
 ASSUME_COMPLETE="true"\
 MODEL_NAME=$MODEL_NAME\
 PAD_TOKEN=$PAD_TOKEN\
 OOV_TOKEN=$OOV_TOKEN\
 BATCH_SIZE=$BATCH_SIZE\
 VALIDATION_BATCH_SIZE=$BATCH_SIZE\
 RANDOM_SEED=$RANDOM_SEED\
 DROPOUT=$DROPOUT\
 LR=$LR\
 NUM_EPOCHS=$NUM_EPOCHS\
 PRIOR_WEIGHT=0.0\
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN

# # Conll german
# # tacl-eer_deu_ee_bcl-bert
bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=3.235.101.13\
#  PRIVATE_IP=172.31.3.32\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=deu\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  LANG_DIR=data/conll2003/deu\
#  TRAIN_DATA=entity.train-docs$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=entity.dev-docs.jsonl\
#  TEST_DATA=entity.test-docs.jsonl\
#  VOCAB_PATH=data/conll2003/mayhew-entity.vocab\
#  BINARY_VOCAB_PATH=data/conll2003/mayhew-binary-entity.vocab\
#  VECTORS_PATH=data/vectors/fasttext.deu.300.vec


# # Conll spanish
# # tacl-eer_esp_ee_bcl-bert
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=3.231.59.126\
#  PRIVATE_IP=172.31.5.152\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=esp\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  LANG_DIR=data/conll2003/esp\
#  TRAIN_DATA=entity.train-docs$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=entity.dev-docs.jsonl\
#  TEST_DATA=entity.test-docs.jsonl\
#  VOCAB_PATH=data/conll2003/mayhew-entity.vocab\
#  BINARY_VOCAB_PATH=data/conll2003/mayhew-binary-entity.vocab\
#  VECTORS_PATH=data/vectors/fasttext.esp.300.vec


# Conll dutch
# tacl-eer_ned_ee_bcl-bert
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=3.239.38.101\
#  PRIVATE_IP=172.31.10.194\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=ned\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  LANG_DIR=data/conll2003/ned\
#  TRAIN_DATA=entity.train-docs$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=entity.dev-docs.jsonl\
#  TEST_DATA=entity.test-docs.jsonl\
#  VOCAB_PATH=data/conll2003/mayhew-entity.vocab\
#  BINARY_VOCAB_PATH=data/conll2003/mayhew-binary-entity.vocab\
#  VECTORS_PATH=data/vectors/fasttext.ned.300.vec


# # Ontonotes english
# # tacl-eer_eng-o_ee_bcl-bert
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=3.231.59.208\
#  PRIVATE_IP=172.31.14.155\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=eng-o\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  LANG_DIR=data/ontonotes5/processed_docs/english\
#  TRAIN_DATA=train$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=dev.jsonl\
#  TEST_DATA=test.jsonl\
#  VOCAB_PATH=data/ontonotes5/processed_docs/mayhew-vocab\
#  BINARY_VOCAB_PATH=data/ontonotes5/processed_docs/mayhew-binary-vocab\
#  VECTORS_PATH=data/vectors/glove.6B.50d.txt


# Ontonotes chinese
# tacl-eer_chi_ee_bcl-bert
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=3.239.162.152\
#  PRIVATE_IP=172.31.2.107\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=chi\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  LANG_DIR=data/ontonotes5/processed_docs/chinese\
#  TRAIN_DATA=train$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=dev.jsonl\
#  TEST_DATA=test.jsonl\
#  VOCAB_PATH=data/ontonotes5/processed_docs/mayhew-vocab\
#  BINARY_VOCAB_PATH=data/ontonotes5/processed_docs/mayhew-binary-vocab\
#  VECTORS_PATH=data/vectors/fasttext.chi.300.vec


# Ontonotes arabic
# tacl-eer_ara_ee_bcl-bert
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=3.235.15.159\
#  PRIVATE_IP=172.31.6.176\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=ara\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  LANG_DIR=data/ontonotes5/processed_docs/arabic\
#  TRAIN_DATA=train$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=dev.jsonl\
#  TEST_DATA=test.jsonl\
#  VOCAB_PATH=data/ontonotes5/processed_docs/mayhew-vocab\
#  BINARY_VOCAB_PATH=data/ontonotes5/processed_docs/mayhew-binary-vocab\
#  VECTORS_PATH=data/vectors/fasttext.ara.300.vec
 

