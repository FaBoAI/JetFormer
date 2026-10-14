no_wall r2(作り直し、2026-10-14、学習の段 no_wall r2)のスクリプト。報告: jetformer_reports/sim_rl_modes_20261013/train_no_wall.md
  run_nw2.mjs / verify_nw2.mjs / own_geom.mjs  確認 r1(rl/check_nw)の写し。NW2_RT(runtime のフォルダー)・NW2_POLICY・NW2_SHIELD で替えられる。
                                               組: VAL(検証の組)・G16・ABL・CORNER(OFF)・FNOM・FDR(OFF)・LOWMU・SELN・SELD・<組>OFF。試験の組(REPLAY)は走らせない
  rt/      シールド v3 の開発の写し(最後は rl/runtime と同じ)。rt_v2/ = v2 の runtime(図の比べ)。rt_v3b/・rt_v3c/ = 比べた v3 の版(deviations.md 4、採ったのは v3c)
  compare_sets.py / build_sets_json.py         組の集計(救った・起こした衝突)
  select_r2.py / after_r2.sh                   チェックポイントの選び・確認・試験(1 回だけ)・書き出し
  write_fid_slow.mjs                           シールドが働く F-ID の列(ゲート 16・舵 0.8・遅延 1)
  line_compare.py / traces_fig.mjs / make_figures_r2.py   線の比べ・図
  ident_probe.mjs / quick_id.mjs / g16quick.mjs / dump1.mjs / dump2.mjs   診断
