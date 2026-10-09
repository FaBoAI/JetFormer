#!/bin/bash
# JetFormer 収集管理画面を起動する(http://<jetson>:8080/)
cd "$(dirname "$0")/.." || exit 1
exec /home/jetson/miniconda/envs/jetracer/bin/python -m jetformer.web.server "$@"
