// 試験の共通部分(2026-10-13、sim_rl_modes)
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
export const EXP = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const FASTEST_POLICY = '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/rl/best/main/policy.json';
export const FASTEST_SHA = '7c43cb53cb24f7951bd4640b7bccb71f34ddbcc43599597aa75248aace6153ee';
export const readJson = p => JSON.parse(fs.readFileSync(p, 'utf8'));
export const sha256File = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
export const fastestWeights = () => readJson(FASTEST_POLICY);
// 79 次元の重みを obs_dim へ 0 で広げた mode 方策(試験用。出力は元と同じ)
export function widenWeights(w, dim, wrapper, mode) {
  const L0 = w.layers[0], nOut = L0.n_out, nIn = L0.n_in, wn = [];
  for (let o = 0; o < nOut; o++) { for (let i = 0; i < nIn; i++) wn.push(L0.w[o * nIn + i]); for (let i = nIn; i < dim; i++) wn.push(0); }
  return {...w, schema: 'jf-sim-rl-mode-policy/v1', obs_dim: dim, obs_wrapper: wrapper, mode, obs_mean: [...w.obs_mean, ...Array(dim - nIn).fill(0)],
    obs_std: [...w.obs_std, ...Array(dim - nIn).fill(1)], layers: [{...L0, n_in: dim, w: wn}, ...w.layers.slice(1)]};
}
