// RL の方策(決定的)を JS で計算する(2026-10-13、RL の学習の段)
//
// Node の評価(rl/evaluate_policy.mjs)とブラウザの差し込みモジュール(記録の段)が同じこのファイルを使う。
// そのため node: の import や Buffer・process を前提にしない(純粋な ESM。process はあれば相対パスの解決にだけ使う)。
// 乱数・時刻も使わない。
//
// 重み(rl/ppo.py export_policy_json が書く JSON、schema rl-bestlap-policy/v1):
//   obs_mean・obs_std(書き出した時点の移動平均・分散で固定)・obs_clip(5)
//   layers: [{n_in, n_out, act: 'tanh' | 'linear', w: 行優先(出力 × 入力)に平らにした値, b}]
// 計算: x = clip((obs − mean) / std, ±5) → 256 → tanh → 256 → tanh → 2(線形)→ [−1, 1] に切る(展開時は平均)。
// 計算は float64(PyTorch の float32 との差は 1 万状態で ≤ 1e-4 を試験で確かめる)。
//
//   import {createPolicy, loadPolicy} from './policy_forward.mjs';
//   const act = createPolicy(weightsJson);        // act(obs Float64Array(79)) → [a0, a1]
//   const fn = await loadPolicy('/path/policy.json');   // rl/evaluate_policy.mjs の --policy の形
export const POLICY_SCHEMA = 'rl-bestlap-policy/v1';

export function createPolicy(weights) {
  if (!weights || weights.schema !== POLICY_SCHEMA) throw new Error('policy_forward: 重みの schema が違います: ' + weights?.schema);
  const nObs = weights.obs_dim, clip = weights.obs_clip, aClip = weights.action_clip ?? 1;
  const mean = Float64Array.from(weights.obs_mean), std = Float64Array.from(weights.obs_std);
  if (mean.length !== nObs || std.length !== nObs) throw new Error('policy_forward: 観測の正規化の長さが違います');
  const layers = weights.layers.map((L, k) => {
    if (L.w.length !== L.n_in * L.n_out || L.b.length !== L.n_out) throw new Error(`policy_forward: 層 ${k} の形が違います`);
    return {nIn: L.n_in, nOut: L.n_out, tanh: L.act === 'tanh', w: Float64Array.from(L.w), b: Float64Array.from(L.b), out: new Float64Array(L.n_out)};
  });
  if (layers[0].nIn !== nObs) throw new Error('policy_forward: 入力の次元が観測と違います');
  const x0 = new Float64Array(nObs);
  const act = (obs, out = [0, 0]) => {
    for (let i = 0; i < nObs; i++) {
      const v = (obs[i] - mean[i]) / std[i];
      x0[i] = v > clip ? clip : v < -clip ? -clip : v;
    }
    let x = x0;
    for (const L of layers) {
      const {nIn, nOut, w, b, out: y} = L;
      for (let o = 0; o < nOut; o++) {
        let s = b[o];
        const base = o * nIn;
        for (let i = 0; i < nIn; i++) s += w[base + i] * x[i];
        y[o] = L.tanh ? Math.tanh(s) : s;
      }
      x = y;
    }
    for (let k = 0; k < out.length; k++) out[k] = x[k] > aClip ? aClip : x[k] < -aClip ? -aClip : x[k];
    return out;
  };
  act.meta = weights.meta ?? {};
  return act;
}

// 重みの JSON を読み込む(Node では絶対・相対のパス、ブラウザでは URL)。import の JSON 属性を使うので fs は要らない
export async function loadWeights(arg) {
  if (!arg) throw new Error('policy_forward: 重みの JSON のパス(--policy-arg)が要ります');
  let url;
  if (/^[a-z][a-z0-9+.-]*:/i.test(arg)) url = arg;
  else if (typeof process !== 'undefined' && process.cwd) url = new URL(arg, 'file://' + process.cwd().replace(/\/?$/, '/')).href;
  else url = new URL(arg, import.meta.url).href;
  const mod = await import(url, {with: {type: 'json'}});
  return mod.default;
}

// rl/evaluate_policy.mjs の形: loadPolicy(arg) → (obs, env) => [a0, a1]
export async function loadPolicy(arg) {
  const act = createPolicy(await loadWeights(arg));
  const fn = obs => act(obs, [0, 0]);
  fn.meta = act.meta;
  return fn;
}
