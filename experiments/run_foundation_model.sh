#!/bin/bash
# Foundation-model side of the benchmark (Chronos-2 zero-shot, adapter selection and seeds, checks). All training uses data up to 2021; selection on 2022.
set -e
PY=env/.venv/bin/python
B=repo/bench
$PY $B/run_chronos.py --data data --out results/preds --name zeroshot
$PY $B/run_chronos.py --data data --out results/val --name zeroshot --wells train --period val
for LR in 1e-4 3e-4; do for ST in 1000 3000; do
  D=artifacts/grid/lr${LR}_st${ST}
  $PY $B/train_lora.py --data data --out $D --steps $ST --lr $LR --end 2021-12-31 --seed 0
  $PY $B/run_chronos.py --data data --out results/val --name lr${LR}_st${ST} --adapter $D/trainer/adapter --wells train --period val
done; done
SEL=$(cd $B && ../../$PY select_lora.py ../../results/val | ../../$PY -c "import json,sys;d=json.load(sys.stdin);assert d['beats_zeroshot'];print(d['selected'])")
LR=$(echo $SEL | sed -E 's/lr([^_]+)_st.*/\1/'); ST=$(echo $SEL | sed -E 's/.*_st([0-9]+)/\1/')
echo "selected $SEL lr=$LR steps=$ST"
mkdir -p artifacts/final
cp -r artifacts/grid/$SEL artifacts/final/seed0
for S in 1 2; do $PY $B/train_lora.py --data data --out artifacts/final/seed$S --steps $ST --lr $LR --end 2021-12-31 --seed $S; done
for S in 0 1 2; do $PY $B/run_chronos.py --data data --out results/preds --name gwfmkr_s$S --adapter artifacts/final/seed$S/trainer/adapter; done
$PY $B/scenario_check.py --data data --out results --name zeroshot
$PY $B/scenario_check.py --data data --out results --name gwfmkr --adapter artifacts/final/seed0/trainer/adapter
for N in 10 40 100; do for S in 1 2; do
  [ $N -eq 100 ] && [ $S -eq 2 ] && continue
  D=artifacts/scale/n${N}_s${S}
  $PY $B/train_lora.py --data data --out $D --steps $ST --lr $LR --end 2021-12-31 --n-wells $N --subset-seed $S --seed 0
  $PY $B/run_chronos.py --data data --out results/preds --name scale_n${N}_s${S} --adapter $D/trainer/adapter --modes A
done; done
echo done
