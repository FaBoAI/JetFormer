// 記録 D3 の差し込み口(2026-10-14、sim_dart_20261014)。コピーした dist の d3.html の import map で、app.js が読む './rc-speed-controller.js' をこのファイルに替える。
// 中身は dart-teacher.js の makeSlot。ユーザーのファイル(app.js・sim-core.js・収集器)は変えない。
import * as original from '../rc-speed-controller.js';
import WEIGHTS from './rt/weights_margin.mjs';
import {SALTS} from './dart_noise.mjs';
import {K_TABLE} from './plan_D3.mjs';
import {makeSlot} from './dart-teacher.js';
export * from '../rc-speed-controller.js';

const slot = makeSlot({dataset: 'D3', kind: 'margin', weights: WEIGHTS, noiseSpec: 'N_margin', salt: SALTS.D3, kTable: K_TABLE, original});
export const assertRcTeacherContract = slot.assertRcTeacherContract;
export const createRcTrainingTeacher = slot.createRcTrainingTeacher;
export const DART_TEACHER_METADATA = slot.METADATA;
