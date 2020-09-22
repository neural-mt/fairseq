#!/bin/bash
# Template

set -x
set -e

GITDIR=$1
DATADIR=$2
OUTDIR=$3
DROPOUT=$4
SRC=fin
TGT=enu

ls $GITDIR
ls $DATADIR

FAIRSEQDIR=$GITDIR
pip install -U torch==1.4.0 torchvision==0.5.0 --user
git clone https://github.com/NVIDIA/apex
cd apex
pip install --user -v --no-cache-dir --global-option="--cpp_ext" \
    --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" \
    --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
cd $FAIRSEQDIR
# pip install --editable . --prefix=$HOME/.local
ls $FAIRSEQDIR
CWD=$PWD


echo '++++++ Cuda/GPU Check+++++++'
nvidia-smi
cat /usr/local/cuda/version.txt
nvcc --version
echo $LD_LIBRARY_PATH
export PATH=/home/viraunak/.local/bin:$PATH
# conda list


echo '++++++ Philly Check +++++++'
echo 'PT_CODE_DIR: $GITDIR'
echo 'PT_DATA_DIR: $DATADIR'
echo 'PT_OUTPUT_DIR: $OUTDIR'


echo '++++++ Python/Python library Check +++++++'
python -V
python -c 'import torch; print("torch: ",torch.__version__, torch.__file__)'
python -c 'import torchvision; print("torchvision: ", torchvision.__version__, torchvision.__file__)'
python -c 'import fairseq; print("fairseq: ",fairseq.__version__, fairseq.__file__)'
python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'


echo '++++++ Fairseq training starts +++++++'
CKPTDIR=$OUTDIR/checkpoint
mkdir -p $CKPTDIR
DATABIN=$DATADIR/data-bin
#mkdir -p $DATABIN

# 1. Create Binary data --> Done Locally
#python $FAIRSEQDIR/preprocess.py \
#       --source-lang $SRC --target-lang $TGT \
#       --trainpref $DATADIR/bpe/train \
#       --validpref $DATADIR/bpe/valid \
#       --testpref $DATADIR/bpe/test \
#       --destdir $DATABIN \
#       --srcdict $DATABIN/dict.$SRC.txt \
#       --tgtdict $DATABIN/dict.$TGT.txt

# 2. Training a model
MAX_TOKENS=3584
BATCH_SIZE=512 # 512 -> 1024 -> 2048
DROPOUT=0.1
### Optimization
MAX_EPOCH=120 # Doubled from 80
# MAX_UPDATE=120000
CLIP_NORM=0  # 25 (default)
# UPDATE_FREQ=1  # default; e.g. 5, 3, 2
#                  (update every {5,3,2} batches in epoch i={5,3,2}) 
OPTIMIZER=adam
LR=1e-3  # 0.25 (default)
# MOMENTUM=0.99  # default
WEIGHT_DECAY=0.0  # default (0.0)
# Note: -- Optimizer (adam) --
# ADAM_BETAS=(0.9, 0.999)  # default
# ADAM_EPS=1e-8 (epsilon for Adam optimizer)
# Learning rate scheduler
# MIN_LR=1e-7  # 1e-5 (default)
# Note: in fairseq/optim/lr_scheduler/cosine_lr_scheduler.py,
#       ``args.min_lr`` is not used.
#       49     self.min_lr = args.lr[0]
#       Rather, args.min_lr used as the minimum lr to stop training!
#       104    while lr > args.min_lr and ...
# LR_SHRINK=0.1  # 0.1 (default)

### Learning rate scheduler (inverse_sqrt)
WARMUP_UPDATES=4000  # warmup the learning rate linearly for the first N updates
WARMUP_INIT_LR=1e-7  # initial learning rate during warmup phase to args.lr

# Fairseq params
#ARCH=prod_model_fin_en
ARCH=transformer_vaswani_wmt_en_fr_big
SEED=1234

python $FAIRSEQDIR/train.py $DATABIN --arch $ARCH \
       --source-lang $SRC --target-lang $TGT --task translation \
       --share-all-embeddings \
       --optimizer adam --adam-betas '(0.9, 0.98)' \
       --clip-norm ${CLIP_NORM} --lr-scheduler inverse_sqrt --lr ${LR} \
       --warmup-updates ${WARMUP_UPDATES} --warmup-init-lr ${WARMUP_INIT_LR} \
       --weight-decay ${WEIGHT_DECAY} --dropout ${DROPOUT} \
       --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
       --max-tokens ${MAX_TOKENS} --batch-size ${BATCH_SIZE} \
       --save-dir ${CKPTDIR} --log-interval 10 --seed $SEED --update-freq 16 \
       --max-epoch ${MAX_EPOCH} --fp16 | tee ${CKPTDIR}/log_train.txt
