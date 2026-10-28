#!/bin/bash
# fix8 係 safety-proposal: pB・pC の試験(PID 766972)が終わってから pA の試験を流し直す(10/28 02:33 に別の係が drive.py の説明の文を
# 足し、最初の pA の試験はその途中で drive.py を読む試験が落ちたため)
D=/home/jetson/jetformer/experiments/20261026_fix/fix8/proposal
while kill -0 766972 2>/dev/null; do sleep 20; done
$D/run_tests10b.sh pA
