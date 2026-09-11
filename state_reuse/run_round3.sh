#!/bin/bash
# Round 3 (ROUND3_PREREG.md): mutation/relocation tolerance of step-0 state reuse.
#   START=50 N=25 CUDA_VISIBLE_DEVICES=0 bash run_round3.sh     (second GPU: START=75 N=25)
# Env: START N CTX TYPES POS STEPS OUT
set -e
cd "$(dirname "$0")"
OUT=${OUT:-results/round3}; mkdir -p "$(dirname "$OUT")"
python run_round3.py --start ${START:-50} --n ${N:-50} --ctx ${CTX:-2048} \
  --types ${TYPES:-edit,insert,delete,reorder} --pos ${POS:-10,25,50,75} \
  --steps ${STEPS:-64} --gen_length 128 --block_length 32 --k ${K:-32} \
  --out $OUT 2>&1 | tee -a $OUT.log
python analyze_round3.py $OUT
