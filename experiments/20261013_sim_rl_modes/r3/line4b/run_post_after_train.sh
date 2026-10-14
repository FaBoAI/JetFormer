#!/bin/bash
# 学習の段(run_line_r4b.sh)の「終わり」を待ってから、post_line_r4b.sh を jf-heavy(lock none、nice 15)で動かす(2026-10-14 夜、line r4b)
C=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/line_r4b/train/logs/chain.log
until grep -q "終わり" "$C"; do sleep 20; done
exec /home/jetson/bin/jf-heavy --mem 1800M --lock none --need-mb 900 -- nice -n 15 bash /home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/line4b/post_line_r4b.sh
