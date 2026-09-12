#!/bin/bash
# Round 4a' (ROUND4AP_PREREG.md): freezing without mutation.
#   CUDA_VISIBLE_DEVICES=0 START=50 N=25 bash run_round4ap.sh ; CUDA_VISIBLE_DEVICES=1 START=75 N=25 bash run_round4ap.sh
set -e
cd "$(dirname "$0")"
OUT=${OUT:-results/round4ap}; mkdir -p "$(dirname "$OUT")"
python run_round4ap.py --start ${START:-50} --n ${N:-50} --pos ${POS:-10,50,90} --conds ${CONDS:-identity,all_but_dep,dep_only,dep_fresh_num} \
  --steps ${STEPS:-128} --gen_length ${GEN:-256} --block_length 32 --k ${K:-32} --out $OUT 2>&1 | tee -a $OUT.log
python analyze_round4ap.py $OUT
