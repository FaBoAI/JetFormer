// DAgger の記録の差し込み口(2026-10-15、sim_diverse_train_20261015)。コピーした dist の dagger.html の import map で、app.js が読む
// './rc-speed-controller.js' をこのファイルに替える。中身は dagger-teacher.js の makeSlot。ユーザーのファイル(app.js・sim-core.js・収集器)は変えない。
import * as original from '../rc-speed-controller.js';
import {makeSlot} from './dagger-teacher.js';
export * from '../rc-speed-controller.js';

const slot = makeSlot({original});
export const assertRcTeacherContract = slot.assertRcTeacherContract;
export const createRcTrainingTeacher = slot.createRcTrainingTeacher;
export const DAGGER_TEACHER_METADATA = slot.METADATA;
