#!/bin/bash
# 反証 8-2 回目・係 rehearsal(8-1 の写し。作り物の家の場所だけ変えた): 手順書 8.1 の休みの塊(```bash の 2 つ目)と 4 章 10 の car_check・8.2 の ⑥〜⑧ を、手順書の文から抜き出して
# そのまま流す。HOME を作り物の家(build_fake.py <日の形>)に替える = ~/jetformer・~/jetformer_reports が作り物を指す。
# 本物の data/raw・labels・jetformer_reports には書かない。python3 は jetracer の環境(Jetson の対話のシェルと同じ)。
#   bash night_rehearsal.sh <日の形> [8.1|cc|8.2]
# car_check の --latest は「今日」(サーバーの時計)の記録を見るので、10/28 のこの稽古では --day 20261103 を足す(手順書の文との違いはそこだけ)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
KIND=${1:-normal}; WHAT=${2:-8.1}
RB=/home/jetson/jetformer_reports/fix_20261026/runbook_20261103.md
FH=/home/jetson/jetformer/experiments/20261026_fix/fix8/tmp_rehearsal_r2/$KIND/home
[ -d "$FH/jetformer" ] || { echo "作り物の家がありません: $FH(build_fake.py $KIND)" >&2; exit 2; }
export HOME="$FH" PATH="/home/jetson/miniconda/envs/jetracer/bin:$PATH" TZ=Asia/Tokyo PYTHONDONTWRITEBYTECODE=1
OUT="$HERE/out/night_${KIND}_${WHAT}.txt"
blk() {  # 手順書の n 番目の ```bash の塊
  python3 -I -c "import re,sys; b=re.findall(r'\`\`\`bash\n(.*?)\`\`\`', open(sys.argv[1],encoding='utf-8').read(), re.S); print(b[int(sys.argv[2])])" "$RB" "$1"
}
case "$WHAT" in
  8.1)
    B=$(blk 4)            # 0: 3 章の curl・1: 3.1・2: 4 章 10 の car_check・3: 5 章の curl・4: 8.1・5: 8.2
    printf '%s\n' "$B" | head -3 >&2
    { echo "### 手順書 8.1 の塊(そのまま)"; printf '%s\n' "$B"; echo "### 出力"; bash -c "$B"; echo "### 終了コード $?"; } > "$OUT" 2>&1
    ;;
  cc)
    B=$(blk 2)
    C="$B --day 20261103"
    { echo "### 手順書 4 章 10: $B(稽古では --day 20261103 を足す)"; bash -c "$C"; echo "### 終了コード $?"; } > "$OUT" 2>&1
    ;;
  8.2)
    B=$(blk 5 | sed 's/<LiPo チェッカーの値>/8.25/')      # 紙の値の置き場所(手順書は <…>)に 8.25 を入れる
    { echo "### 手順書 8.2 の塊(そのまま。⑦ は装置を開かない GET。作り物の家なので fw_vbat_check の URL は既定のまま)"; printf '%s\n' "$B";
      echo "### 出力"; bash -c "$B"; echo "### 終了コード $?"; } > "$OUT" 2>&1
    ;;
esac
echo "$OUT"
