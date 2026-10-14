#!/usr/bin/env node
// 速さいろいろ r2 の書き出し(train/speed/r2/export/policy.json)を、ブラウザと同じ組 web/rl/ ではなく rl/runtime/mode_runtime.mjs で読めることの確かめ
// (2026-10-14 学習の段 speed r2)。STD のフレーム 0 の指令・p・metadata と、初期化元が止まった開始(検証の P36)を元のコアで走らせる。
import fs from 'node:fs';
import crypto from 'node:crypto';
import {createModeTeacher} from '../runtime/mode_runtime.mjs';
import {runModeEpisode, BASE_WORLD} from '../episode.mjs';
import {activeWorldFor, simConfig} from '../../snapshot/rl/env_core.mjs';
import * as core from '../../snapshot/sim/dist/sim-core.js';

const pp = process.argv[2], W = JSON.parse(fs.readFileSync(pp, 'utf8')), sha = crypto.createHash('sha256').update(fs.readFileSync(pp)).digest('hex');
const {world, variation} = activeWorldFor(BASE_WORLD, 0, 0, 0);
for (const cv of [1.0, .5]) {
  const cfg = simConfig({episodeMode: 'continuous', seed: 1}), sim = core.createSimulation(world, cfg);
  let snap = sim.reset({seed: 1});
  const t = createModeTeacher(world, JSON.parse(JSON.stringify(snap.config)), variation, {mode: 'speed', weights: W, params: {c_v: cv}, policySha256: sha});
  let maxCmd = 0;
  for (let f = 0; f < 120; f++) { const a = t.action(snap); maxCmd = Math.max(maxCmd, a.speed); if (f === 0) console.log(`c_v ${cv}: フレーム 0 の p ${JSON.stringify(a.info.jf.p)}`); snap = sim.step(a); }
  console.log(`c_v ${cv}: 最初の 120 フレームの速さの指令の最大 ${maxCmd.toFixed(4)} m/s(上限 ${(4.5 * cv).toFixed(3)})・metadata ${JSON.stringify({id: t.metadata.id, obs_wrapper: t.metadata.obs_wrapper, speed_cap_mps: t.metadata.speed_cap_mps, policy_sha256: t.metadata.policy_sha256.slice(0, 12), line_profile: t.metadata.line_profile ?? null})}`);
}
const r = await runModeEpisode({mode: 'speed', weights: W, policySha256: sha, start: {gate: 3, left_offset_m: 0.15197886501041508, left_yaw_deg: -11.14464, seed: 805036}, params: {c_v: .875}});
console.log(`初期化元が止まった開始(P36・c_v 0.875): ${r.reason}・周 ${r.laps.map(l => l.elapsed.toFixed(2)).join(', ')}・すき間 ${(r.min_clearance * 100).toFixed(2)} cm`);
