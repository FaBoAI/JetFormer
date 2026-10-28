#!/bin/bash
# fix8 係 safety-proposal: batch 1(run10_batch1)が終わるのを待ってから batch 2(前からのくみ・100 Hz の dip・でたらめ)を流す
D=/home/jetson/jetformer/experiments/20261026_fix/fix8/proposal
while kill -0 744401 2>/dev/null; do sleep 20; done
cd $D && ./run10.sh burst "brake:-" align_orig_50 align_orig_100 align_human_50 align_human_100 forms hold_orig_50 hold_orig_100 \
  hold_human_50 hold_human_100 rshort dip_norm_100 dip_skip_100 > $D/out/run10_batch2.log 2>&1
cd /home/jetson/jetformer
for seed in 8801 8802; do
  PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
    $D/fuzz10.py 150 $seed > $D/out/fuzz10_$seed.log 2>&1 &
done
wait
echo all batches done >> $D/out/run10_batch2.log
