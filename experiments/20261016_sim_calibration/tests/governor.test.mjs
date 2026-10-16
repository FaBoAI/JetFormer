// 試験 T6: 上限の層の JS(calib_runtime.mjs の governorStep)が、Python の安全層 v1(jetformer/safety_caps.py の cap_step)と
// 同じ入力で同じ出力を出す(4 つの設定 × 3,000 フレーム = 12,000 通り)。あわせて不変条件(出力 ≤ 入力・ブレーキは R1_app だけ・
// 始まりは 1 m/s 以上・往復は 0.5 秒以上あく)を JS の側でも確かめる。
//   python tests/t6_make_vectors.py && node --test tests/governor.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {governorStep, governorInitialState} from '../calib_runtime.mjs';

const V = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_calibration_20261016/tests/t6_vectors.json', 'utf8'));
const dec = x => x === 'inf' ? Infinity : x === '-inf' ? -Infinity : x;

for (const [name, set] of Object.entries(V.sets)) {
  test(`T6 ${name}: Python の cap_step と同じ出力(${set.rows.length} フレーム)`, () => {
    let st = governorInitialState();
    let lastEnd = -1e9, braking = false, n = 0;
    for (const [t, zone, v, u, steer, eu, emode, ecap, estart] of set.rows) {
      const r = governorStep(st, t, zone, v, u, set.params, steer);
      st = r.state;
      assert.equal(r.out.mode, emode, `mode t=${t}`);
      assert.ok(Math.abs(r.out.u - eu) <= 1e-12, `u t=${t}: ${r.out.u} ${eu}`);
      const c = dec(ecap);
      assert.ok(c === r.out.cap_eff || Math.abs(c - r.out.cap_eff) <= 1e-12, `cap t=${t}`);
      assert.equal(r.out.brake_start, estart);
      assert.ok(r.out.u <= u + 1e-12);
      if (r.out.mode === 'brake') {
        assert.ok(set.params.brake_zones.includes(zone));
        if (r.out.brake_start) { assert.ok(v >= 1.0); assert.ok(t - lastEnd >= .5 - 1e-9); }
        braking = true;
      } else if (braking) { braking = false; lastEnd = t; }
      n++;
    }
    assert.equal(n, set.rows.length);
  });
}
