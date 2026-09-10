#!/bin/bash
# Round 2 GPU run.  Prompts = gsm8k_test200[START:START+N] (default 50..100), history pool = [100:200].
#   bash run_round2.sh                                  # full preregistered run
#   START=50 N=25 CTX=0,256,512 bash run_round2.sh      # split by prompts and/or context lengths
# Several splits may write to the same OUT (records are appended); run analyze_round2.py once at the end.
set -e
cd "$(dirname "$0")"
OUT=${OUT:-results/round2}; mkdir -p "$(dirname "$OUT")"
python run_round2.py --start ${START:-50} --n ${N:-50} --ctx ${CTX:-0,256,512,1024,2048} \
  --steps ${STEPS:-64} --gen_length ${GEN:-128} --block_length 32 \
  --ks ${KS:-4,8,12,16,20,24,28,32} --out $OUT 2>&1 | tee -a $OUT.log
python analyze_round2.py $OUT
