#!/bin/bash
# Round 4a (ROUND4A_PREREG.md): dependency-changing mutations.
#   CUDA_VISIBLE_DEVICES=0 START=50 N=50 bash run_round4a.sh
set -e
cd "$(dirname "$0")"
OUT=${OUT:-results/round4a}; mkdir -p "$(dirname "$OUT")"
python run_round4a.py --start ${START:-50} --n ${N:-50} --pos ${POS:-10,50} --conds ${CONDS:-ctrl_other,ctrl_name,dep_number,dep_move} \
  --steps ${STEPS:-64} --gen_length 128 --block_length 32 --k ${K:-32} --out $OUT 2>&1 | tee -a $OUT.log
python analyze_round4a.py $OUT
