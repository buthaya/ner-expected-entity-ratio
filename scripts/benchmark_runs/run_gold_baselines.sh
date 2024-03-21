# Benchmark Experiment: Gold mle baseline on all datasets
RUN_JUPYTER=true
RUN_TENSORBOARD=true
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


# Conll english
# tacl-eer_eng-c_gold_mle
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=3.90.34.198\
 PRIVATE_IP=172.31.32.220\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=eng-c\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 BASE_CONFIG=experiments/supervised_tagger.jsonnet\
 ASSUME_COMPLETE=$ASSUME_COMPLETE\
 LANG_DIR=data/conll2003/eng\
 TRAIN_DATA=entity.train-docs.jsonl\
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
 PRIOR_TYPE=$PRIOR_TYPE\
 PRIOR_WEIGHT=$PRIOR_WEIGHT\
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN
 


# Conll german
# tacl-eer_deu_gold_mle
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=3.80.234.112\
 PRIVATE_IP=172.31.36.16\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=deu\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 BASE_CONFIG=experiments/supervised_tagger.jsonnet\
 ASSUME_COMPLETE=$ASSUME_COMPLETE\
 LANG_DIR=data/conll2003/deu\
 TRAIN_DATA=entity.train-docs.jsonl\
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
 PRIOR_TYPE=$PRIOR_TYPE\
 PRIOR_WEIGHT=$PRIOR_WEIGHT\
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN



# Conll spanish
# tacl-eer_esp_gold_mle
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=54.162.169.42\
 PRIVATE_IP=172.31.41.154\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=esp\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 BASE_CONFIG=experiments/supervised_tagger.jsonnet\
 ASSUME_COMPLETE=$ASSUME_COMPLETE\
 LANG_DIR=data/conll2003/esp\
 TRAIN_DATA=entity.train-docs.jsonl\
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
 PRIOR_TYPE=$PRIOR_TYPE\
 PRIOR_WEIGHT=$PRIOR_WEIGHT\
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN


# Conll dutch
# tacl-eer_ned_gold_mle
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=35.153.160.38\
 PRIVATE_IP=172.31.34.82\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=ned\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 BASE_CONFIG=experiments/supervised_tagger.jsonnet\
 ASSUME_COMPLETE=$ASSUME_COMPLETE\
 LANG_DIR=data/conll2003/ned\
 TRAIN_DATA=entity.train-docs.jsonl\
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
 PRIOR_TYPE=$PRIOR_TYPE\
 PRIOR_WEIGHT=$PRIOR_WEIGHT\
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN


# Ontonotes english
# tacl-eer_eng-o_gold_mle
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=3.88.71.105\
 PRIVATE_IP=172.31.44.255\
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


# Ontonotes chinese
# tacl-eer_chi_gold_mle
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=34.235.144.234\
 PRIVATE_IP=172.31.35.163\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=chi\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 BASE_CONFIG=experiments/supervised_tagger.jsonnet\
 ASSUME_COMPLETE=$ASSUME_COMPLETE\
 LANG_DIR=data/ontonotes5/processed_docs/chinese\
 TRAIN_DATA=train.jsonl\
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


# Ontonotes arabic
# tacl-eer_ara_gold_mle
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=34.239.141.209\
 PRIVATE_IP=172.31.35.171\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=ara\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 BASE_CONFIG=experiments/supervised_tagger.jsonnet\
 ASSUME_COMPLETE=$ASSUME_COMPLETE\
 LANG_DIR=data/ontonotes5/processed_docs/arabic\
 TRAIN_DATA=train.jsonl\
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

