# Benchmark Experiment: Raw MLE approach on non-native speaker (nns) datasets
RUN_JUPYTER=false
RUN_TENSORBOARD=false
DATASET_LABEL=nns
METHOD_LABEL=raw-mle
TRAIN_SUFFIX=_r0.5_p0.9
ASSUME_COMPLETE="true"
RANDOM_SEED=0
DROPOUT=0.2
LR=2e-5
NUM_EPOCHS=20
PRIOR_TYPE=null
PRIOR_WEIGHT=0.0
ENTITY_RATIO=0.15
ENTITY_RATIO_MARGIN=0.05


# Conll english
# tacl-eer_eng-c_nns_raw-mle
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=54.162.238.102\
 PRIVATE_IP=172.31.37.223\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=eng-c\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 BASE_CONFIG=experiments/supervised_tagger.jsonnet\
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
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN
 


# Conll german
# tacl-eer_deu_nns_raw-mle
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=54.159.64.71\
 PRIVATE_IP=172.31.35.97\
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
 MODEL_NAME=bert-base-multilingual-cased\
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



# Conll spanish
# tacl-eer_esp_nns_raw-mle
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=54.172.138.17\
 PRIVATE_IP=172.31.40.205\
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
 MODEL_NAME=bert-base-multilingual-cased\
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


# Conll dutch
# tacl-eer_ned_nns_raw-mle
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=54.224.110.141\
 PRIVATE_IP=172.31.42.19\
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
 MODEL_NAME=bert-base-multilingual-cased\
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


# Ontonotes english
# tacl-eer_eng-o_nns_raw-mle
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=3.85.176.171\
 PRIVATE_IP=172.31.33.130\
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


# Ontonotes chinese
# tacl-eer_chi_nns_raw-mle
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=3.91.228.126\
 PRIVATE_IP=172.31.37.149\
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
 MODEL_NAME=bert-base-multilingual-cased\
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


# Ontonotes arabic
# tacl-eer_ara_nns_raw-mle
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=54.159.49.101\
 PRIVATE_IP=172.31.42.61\
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
 MODEL_NAME=bert-base-multilingual-cased\
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

