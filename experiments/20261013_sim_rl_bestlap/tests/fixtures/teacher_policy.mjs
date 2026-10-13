// 試験用の「方策のモジュール」: 観測は使わず、env の正解の状態から RC 教師の操作を計算して a ∈ [−1, 1]² で返す。
// rl/evaluate_policy.mjs の --policy の形(loadPolicy)の確認用。学習した方策ではない。
import {loadRcTeacher} from '../../rl/env_core.mjs';
import {commandToAction} from '../../rl/features.mjs';
export async function loadPolicy() {
  const T = await loadRcTeacher();
  let key = null, teacher = null;
  return (obs, env) => {
    if (key !== env.sim) { key = env.sim; teacher = T.createRcTrainingTeacher(env.world, JSON.parse(JSON.stringify(env.config)), env.variation); }
    const c = teacher.action(env.view), a = commandToAction(c.steering, c.speed, env.speedCap);
    return [a.a0, a.a1];
  };
}
