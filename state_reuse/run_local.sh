#!/bin/zsh
# Sequential local pilot on MPS: one prompt per invocation so partial results are always plottable.
source ~/.venvs/llada/bin/activate
cd "$(dirname "$0")"
OUT=${OUT:-results/pilot_mps}
START=${START:-0}; N=${N:-6}
for ((i=START; i<START+N; i++)); do
  python run_pilot.py --start $i --n 1 --steps ${STEPS:-32} --gen_length ${GEN:-128} --block_length 32 \
    --ks ${KS:-0,8,16,24,32} --sources ${SOURCES:-aligned,final} --resume_steps ${RESUME:-8,16} --out $OUT 2>&1 | grep -v -i warn | grep -v "Loading checkpoint"
done
echo ALL_DONE
