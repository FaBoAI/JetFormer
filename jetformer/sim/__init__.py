"""シミュレーター(CourseLab)を JetFormer の管理画面に統合する

- install.py      ~/sim に置いたシミュレーター本体の ZIP を data/sim/releases/<版>/ に導入する
- env.py          ブリッジ用の Python 環境(data/sim/venv)を準備する
- bridge_runner.py / bridge_proc.py   ブリッジ(ONNX の CPU 推論とデータ保存)を子プロセスとして動かす
- proxy.py        ブリッジの API を /sim/bridge/api/* で中継する
- courses.py / worldhash.py   コースの登録簿(Default・同梱の旧コース・アップロードしたコース)
- datasets.py     ブリッジが集めたデータセットの一覧・取り込み・削除
- web.py          /sim/* と /api/sim/* の経路(server.py が include_router する)

ユーザーのシミュレーターのファイルは変えない(写しを data/sim/releases/ に置き、その写しも変えない)。
"""
