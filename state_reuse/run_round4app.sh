#!/bin/bash
# Round 4a'' (ROUND4APP_PREREG.md): confirmatory block-liveness replication on GSM8K test[200:300].
#   CUDA_VISIBLE_DEVICES=0 START=200 N=50 bash run_round4app.sh ; CUDA_VISIBLE_DEVICES=1 START=250 N=50 bash run_round4app.sh
set -e
cd "$(dirname "$0")"
OUT=${OUT:-results/round4app}; mkdir -p "$(dirname "$OUT")"
python run_round4ap.py --data gsm8k_test_full.json --start ${START:-200} --n ${N:-100} --pos ${POS:-10,90} \
  --conds ${CONDS:-identity,all_but_dep} --steps ${STEPS:-128} --gen_length ${GEN:-256} --block_length 32 --k ${K:-32} \
  --log_attn --out $OUT 2>&1 | tee -a $OUT.log
python analyze_round4app.py $OUT results/round4ap
