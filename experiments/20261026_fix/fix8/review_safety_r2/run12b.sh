#!/bin/bash
# 10/28 10:0x(main session「台本は短めに・場面は 8-1 より少なくてよい・違う角度を優先」)のあとの残り: ESC の模型を選んで流す。
# tb・hm は run12.sh のまま流し切る(1 組 1 行なので止まっても続きから)。
cd /home/jetson/jetformer || exit 1
D=experiments/20261026_fix/fix8/review_safety_r2
PY="/home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I -B"
export PYTHONDONTWRITEBYTECODE=1
SEL=fit,cl,human,fit_lag0.1
( RS82_ESC=fit,cl,human $PY $D/idold12.py 0/1 >> $D/out/run_idold12_0of1.log 2>&1
  RS82_ESC=$SEL $PY $D/exp12.py hb 0/1 >> $D/out/run_exp12_hb_0of1.log 2>&1
  echo "chain A finished $(date)" >> $D/out/run12_workers.log ) &
( RS82_ESC=fit,cl,human $PY $D/exp12c.py 0/1 >> $D/out/run_exp12c_0of1.log 2>&1
  echo "chain B finished $(date)" >> $D/out/run12_workers.log ) &
( while pgrep -f "review_safety_r2/exp12.py tb" > /dev/null; do sleep 30; done
  RS82_ESC=$SEL $PY $D/exp12.py hmh 0/1 >> $D/out/run_exp12_hmh_0of1.log 2>&1
  echo "chain C finished $(date)" >> $D/out/run12_workers.log ) &
wait
echo "run12b all finished $(date)" >> $D/out/run12_workers.log
