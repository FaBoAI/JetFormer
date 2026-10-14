#!/bin/bash
# 昇格の段: 記録の進みに合わせて確かめを走らせる(離脱復活の取り込みが済んだら離脱復活を、プランが終わったら残りと物差し)
set -u
PID=20261015-030258-d27688
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
O=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/promote/record
P=/home/jetson/jetformer/data/sim_rl/plans/$PID.json
cd $EXP
H="/home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 900 --"
st() { python3 -c "import json;p=json.load(open('$P'));print(((p.get('imports') or {}).get('$1') or {}).get('state') or '', ((p.get('imports') or {}).get('$1') or {}).get('dataset') or '')"; }
until [ "$(st recover | cut -d' ' -f1)" = done ] || [ "$(st recover | cut -d' ' -f1)" = error ] || [ -f $O/status ]; do sleep 20; done
read s n <<< "$(st recover)"
if [ "$s" = done ]; then
  echo "$(date -Is) recover $n" >> $O/chain.log
  $H nice -n 15 node r3/promote/replay_check_r4.mjs --dataset /home/jetson/jetformer/data/datasets/$n --out $O/replay/recover.json >> $O/chain.log 2>&1
  $H nice -n 15 /home/jetson/miniconda/envs/jetracer/bin/python -I record/verify_mode_dataset.py /home/jetson/jetformer/data/datasets/$n --out $O/verify/recover.json >> $O/chain.log 2>&1
  $H nice -n 15 node r3/rc4/promote/check_recorded_scenes_rc4.mjs /home/jetson/jetformer/data/datasets/$n --out $O/check_recorded/recover.json >> $O/chain.log 2>&1
  echo "$(date -Is) recover checks done" >> $O/chain.log
fi
until [ -f $O/status ]; do sleep 20; done
echo "$(date -Is) plan finished: $(cat $O/status)" >> $O/chain.log
bash r3/promote/after_record_r3.sh $PID >> $O/after_record.log 2>&1
echo "$(date -Is) after_record exit $?" >> $O/chain.log
