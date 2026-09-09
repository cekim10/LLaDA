#!/bin/bash
# GPU-server run.  Copy the state_reuse/ directory to the server, then:
#   pip install "torch" "transformers==4.38.2" accelerate numpy matplotlib
#   bash run_server.sh
# Full pilot: 50 prompts x 5 deltas, 128 gen tokens, 64 steps, k sweep + resume sweep.
set -e
cd "$(dirname "$0")"
OUT=${OUT:-results/pilot_gpu}; mkdir -p "$(dirname "$OUT")"
python run_pilot.py --start ${START:-0} --n ${N:-50} --steps ${STEPS:-64} --gen_length ${GEN:-128} --block_length 32 \
  --ks ${KS:-0,4,8,12,16,20,24,28,32} --sources ${SOURCES:-aligned,final,first} --resume_steps ${RESUME:-8,16,32,48} \
  --out $OUT 2>&1 | tee -a $OUT.log
python plot.py $OUT
