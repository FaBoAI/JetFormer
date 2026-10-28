#!/bin/bash
# fix8 係 safety-decided: drive.py の説明の文を足したあと、drive.py を読む試験(tests/ の 20 本 + 実験の 5 本)と走行ループのほかの試験 7 本を流す
set -u
cd /home/jetson/jetformer
BT=/home/jetson/jetformer/experiments/20261026_fix/fix8/tmp_safety_decided/pt_all
OUT=/home/jetson/jetformer/experiments/20261026_fix/fix8/decided_safety/out
export PYTHONDONTWRITEBYTECODE=1
READ="tests/test_drive_caps.py tests/test_fix3_r1_1.py tests/test_fix3_r2_1.py tests/test_fix3_r3_1.py tests/test_fix3_r3_2_tools.py
 tests/test_fix4_small.py tests/test_fix7_docs_final.py tests/test_fix7_docs_r2.py tests/test_launch_drive.py tests/test_rec_fix5_safety.py
 tests/test_rec_fix6_safety.py tests/test_rec_fix6_safety_r2.py tests/test_rec_fix7_safety.py tests/test_rec_fix7_safety_r2.py
 tests/test_rec_hold_keep.py tests/test_recovery_back_guard.py tests/test_rl_evaluate.py tests/test_speed_ctrl.py tests/test_ui_fix3_r1_2.py
 tests/test_ui_fix5.py"
EXPT="experiments/20261026_fix/fix/test_fix_launch_procedure.py experiments/20261026_fix/verify/test_verify_launch.py
 experiments/20261026_fix/fix2/test_desk_v3.py experiments/20261026_fix/fix2/safety1/test_probe_safety1.py"
# experiments/20261025_train_v2/c3/test_desk_v2.py はモジュールの終わりで sys.exit する台本の形なので、pytest に入れずに台本として流す(下)。
# verify/test_verify_launch.py は流すたびに verify/out/*.json を書き直す(cam_epoch_us の末だけ違う)→ 流したあと git の HEAD に戻すか見る
LOOP="tests/test_drive.py tests/test_drive_midrun_params.py tests/test_drive_caps_fix.py tests/test_drive_ai_time.py
 tests/test_drive_model_select.py tests/test_speed_ctrl_replay.py tests/test_speed_ctrl_pure.py"
echo "== $(date +%T) drive.py $(sha256sum jetformer/drive.py | cut -c1-16) assist.py $(sha256sum jetformer/assist.py | cut -c1-16)"
/home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -B -m pytest -q -p no:cacheprovider \
  -o tmp_path_retention_policy=failed --basetemp=$BT -rfEs $READ > $OUT/pytest_after_read.txt 2>&1
tail -n 3 $OUT/pytest_after_read.txt
/home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -B -m pytest -q -p no:cacheprovider \
  -o tmp_path_retention_policy=failed --basetemp=${BT}_x -rfEs $EXPT > $OUT/pytest_after_expt.txt 2>&1
tail -n 3 $OUT/pytest_after_expt.txt
(cd experiments/20261025_train_v2/c3 && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I -B \
  test_desk_v2.py > $OUT/script_test_desk_v2.txt 2>&1; echo "exit $?" >> $OUT/script_test_desk_v2.txt)
tail -n 2 $OUT/script_test_desk_v2.txt
git status --short experiments/20261026_fix/verify/out/
/home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -B -m pytest -q -p no:cacheprovider \
  -o tmp_path_retention_policy=failed --basetemp=${BT}_l -rfEs $LOOP > $OUT/pytest_after_loop.txt 2>&1
tail -n 3 $OUT/pytest_after_loop.txt
rm -rf $BT ${BT}_x ${BT}_l
echo "== $(date +%T) end"
