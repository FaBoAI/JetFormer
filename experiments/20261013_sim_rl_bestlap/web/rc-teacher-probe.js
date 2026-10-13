// 差し替えの確認用(何もしない): 元の rc-speed-controller.js をそのまま再エクスポートし、
// アプリ(app.js)が import map 経由でこのモジュールを読んだことだけを window に記録する。
// ユーザーのファイルは変更しない。コピーした dist の rl.html の import map でだけ有効。
import * as original from '../rc-speed-controller.js';
export * from '../rc-speed-controller.js';
const calls={createRcTrainingTeacher:0,actions:0};
globalThis.__rlTeacherProbe={loaded:true,calls,originalExports:Object.keys(original)};
export function createRcTrainingTeacher(world,config,variation={}){
  calls.createRcTrainingTeacher++;
  const teacher=original.createRcTrainingTeacher(world,config,variation);
  const action=teacher.action;
  teacher.action=snapshot=>{calls.actions++;return action(snapshot);};
  return teacher;
}
