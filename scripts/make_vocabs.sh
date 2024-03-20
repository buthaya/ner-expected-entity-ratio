#!/bin/bash 

#SBATCH--job-name=MakeVocab 
#SBATCH--output=MakeVocab%j.out 
#SBATCH--error=MakeVocab%j.err 
#SBATCH--partition=prepost
#SBATCH--nodes=1 
#SBATCH--ntasks=4 
#SBATCH--cpus-per-task=10 
#SBATCH--time=01:00:00 
#SBATCH--qos=qos_gpu-dev 
#SBATCH--hint=nomultithread 
#SBATCH--account=zke@v100 

module purge 
conda deactivate
module load pytorch-gpu/py3/1.12.1
conda activate ner-eer/
set-x

# Precompute vocab files for models so we can control which tags are which index (and for speed)

# python -m ml.cmd.make_vocab roberta-base data/conll2003/eng/entity.train.jsonl data/conll2003/roberta-entity.vocab
# python -m ml.cmd.make_vocab bert-base-multilingual-cased data/conll2003/deu/entity.train.jsonl data/conll2003/mbert-entity.vocab
srun python -m ml.cmd.make_vocab roberta-base data/ontonotes5/processed/english/train.jsonl data/ontonotes5/processed/roberta-entity.vocab
srun python -m ml.cmd.make_vocab bert-base-multilingual-cased data/ontonotes5/processed/chinese/train.jsonl data/ontonotes5/processed/mbert-entity.vocab